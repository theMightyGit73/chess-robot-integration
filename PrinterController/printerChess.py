import serial
import pigpio
import time
import json
import os
import sys
import re
import logging
import traceback
import chess
import chess.engine
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class Color(Enum):
    WHITE = 'white'
    BLACK = 'black'

class PieceType(Enum):
    PAWN = 'pawn'
    KNIGHT = 'knight'
    BISHOP = 'bishop'
    ROOK = 'rook'
    QUEEN = 'queen'
    KING = 'king'


@dataclass
class ChessPiece:
    type: PieceType
    color: Color
    square: str  # e.g., 'e2'

@dataclass
class PrinterConfig:
    """Configuration settings for the printer and gripper."""
    printer_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    servo_pin: int = 13
    z_offset_file: str = "z_offset.json" 
    
    # Servo positions (150° to 175° range)
    gripper_open_pw: int = 2200  # ~150° (fully open)
    gripper_closed_pw: int = 2450  # ~175° (fully closed)
    
    # Movement settings
    gripper_step: int = 5  # µs step per iteration
    gripper_delay: float = 0.005  # seconds
    xy_feedrate: int = 6000  # Increased for faster movement
    z_feedrate: int = 2000   # Increased for faster movement
    
    # Movement limits (in mm)
    max_z_height: float = 200.0
    min_z_height: float = -50.0  # Increased range
    max_xy_distance: float = 220.0
    
    def get_z_offset_path(self):
        """Find the z_offset file by searching in multiple directories."""
        import os
        
        # List of directories to search
        search_dirs = [
            '.',  # Current directory
            os.path.dirname(os.path.abspath(__file__)),  # PrinterController directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),  # Parent directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PrinterController'),  # Named subdirectory
            os.path.expanduser("~"),  # Home directory
        ]
        
        # Try to find the file in any of these directories
        for directory in search_dirs:
            path = os.path.join(directory, self.z_offset_file)
            if os.path.exists(path):
                return path
                
        # If not found, return default path in current directory
        return self.z_offset_file

@dataclass
class ChessBoardConfig:
    """Configuration for chess board positions on printer bed."""
    positions_file: str = "chess_positions.json"

@dataclass
class ChessPieceConfig:
    """Configuration for chess piece settings."""
    pieces_file: str = "piece_settings.json"
    pieces = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

@dataclass
class GripperConfig:
    """Configuration for gripper settings."""
    settings_file: str = "gripper_settings.json"
    
@dataclass
class StorageConfig:
    """Configuration for piece storage positions."""
    storage_file: str = "storage_positions.json"
    positions = ['box_1', 'box_2']  # The storage locations we need to track



class StoragePositionMapper:
    """Maps storage positions on the printer bed."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.positions = {}
        self._load_positions()
    
    def _load_positions(self) -> None:
        """Load storage positions from JSON file."""
        try:
            if Path(self.config.storage_file).exists():
                with open(self.config.storage_file, 'r') as f:
                    self.positions = json.load(f)
                print(f"Loaded {len(self.positions)} storage positions")
        except Exception as e:
            print(f"Error loading storage positions: {str(e)}")
            self.positions = {}
    
    def _save_positions(self) -> None:
        """Save storage positions to JSON file."""
        try:
            with open(self.config.storage_file, 'w') as f:
                json.dump(self.positions, f, indent=2)
            print(f"Saved {len(self.positions)} storage positions")
        except Exception as e:
            print(f"Error saving storage positions: {str(e)}")
    
    def set_position(self, location: str, x: float, y: float, z: float) -> None:
        """Set physical coordinates for a storage location."""
        self.positions[location.lower()] = {'x': x, 'y': y, 'z': z}
        self._save_positions()
    
    def get_position(self, location: str) -> Optional[Dict[str, float]]:
        """Get physical coordinates for a storage location."""
        location = location.lower()
        return self.positions.get(location)
    

class GripperSettingsMapper:
    """Maps and manages gripper settings."""
    
    def __init__(self, config: GripperConfig):
        self.config = config
        self.settings = {
            'open_pw': 2300,
            'closed_pw': 2500,
            'step': 5,
            'delay': 0.005
        }
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load gripper settings from JSON file."""
        try:
            if Path(self.config.settings_file).exists():
                with open(self.config.settings_file, 'r') as f:
                    self.settings = json.load(f)
                print(f"Loaded gripper settings")
        except Exception as e:
            print(f"Error loading gripper settings: {str(e)}")
    
    def _save_settings(self) -> None:
        """Save gripper settings to JSON file."""
        try:
            with open(self.config.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"Saved gripper settings")
        except Exception as e:
            print(f"Error saving gripper settings: {str(e)}")
    
    def get_settings(self) -> Dict[str, Union[int, float]]:
        """Get current gripper settings."""
        return self.settings
    
    def update_settings(self, settings: Dict[str, Union[int, float]]) -> None:
        """Update gripper settings."""
        self.settings.update(settings)
        self._save_settings()


class ChessPieceMapper:
    """Maps chess pieces to their heights and grip settings."""
    
    def __init__(self, config: ChessPieceConfig):
        self.config = config
        self.piece_settings = {}
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load piece settings from JSON file."""
        try:
            if Path(self.config.pieces_file).exists():
                with open(self.config.pieces_file, 'r') as f:
                    self.piece_settings = json.load(f)
                print(f"Loaded {len(self.piece_settings)} piece settings")
        except Exception as e:
            print(f"Error loading piece settings: {str(e)}")
            self.piece_settings = {}
    
    def _save_settings(self) -> None:
        """Save piece settings to JSON file."""
        try:
            with open(self.config.pieces_file, 'w') as f:
                json.dump(self.piece_settings, f, indent=2)
            print(f"Saved {len(self.piece_settings)} piece settings")
        except Exception as e:
            print(f"Error saving piece settings: {str(e)}")
    
    def set_piece_settings(self, piece: str, height: float, grip_pw: int) -> None:
        """Set height and grip settings for a specific piece."""
        self.piece_settings[piece.lower()] = {
            'height': height,
            'grip_pw': grip_pw
        }
        self._save_settings()
    
    def get_piece_settings(self, piece: str) -> Optional[Dict[str, Union[float, int]]]:
        """Get settings for a specific piece."""
        piece = piece.lower()
        return self.piece_settings.get(piece)


class ChessPositionMapper:
    """Maps chess coordinates to physical positions on the printer bed."""
    
    # Add at the beginning of the ChessPositionMapper class
    def __init__(self, config: ChessBoardConfig):
        self.config = config
        self.positions = {}
        self.flipped = False  # Track if board is flipped (black perspective)
        self.logger = logging.getLogger("ChessPositionMapper")
        self._load_positions()
    
    def _load_positions(self) -> None:
        """Load chess positions from JSON file."""
        try:
            if Path(self.config.positions_file).exists():
                with open(self.config.positions_file, 'r') as f:
                    self.positions = json.load(f)
                print(f"Loaded {len(self.positions)} chess positions")
        except Exception as e:
            print(f"Error loading chess positions: {str(e)}")
            self.positions = {}
    
    def _save_positions(self) -> None:
        """Save chess positions to JSON file."""
        try:
            with open(self.config.positions_file, 'w') as f:
                json.dump(self.positions, f, indent=2)
            print(f"Saved {len(self.positions)} chess positions")
        except Exception as e:
            print(f"Error saving chess positions: {str(e)}")
    
    def set_position(self, square: str, x: float, y: float, z: float) -> None:
        """Set physical coordinates for a chess square."""
        self.positions[square.lower()] = {'x': x, 'y': y, 'z': z}
        self._save_positions()
    
    def get_position(self, square: str) -> Optional[Dict[str, float]]:
        """Get physical coordinates for a chess square."""
        square = square.lower()
        if square not in self.positions:
            return None
        return self.positions[square]

    def flip_board(self, flipped=True):
        """Set board orientation: True for black's perspective, False for white's"""
        self.flipped = flipped
        self.logger.info(f"Board orientation set to: {'black perspective' if flipped else 'white perspective'}")

    def transform_square(self, square: str) -> str:
        """Transform square name based on current orientation"""
        square = square.lower()
        if not self.flipped:
            self.logger.debug(f"No transformation needed for {square} (white perspective)")
            return square
            
        # Transform for black's perspective
        file = square[0]  # a-h
        rank = square[1]  # 1-8
        
        # Flip file from a→h, b→g, etc. and rank from 1→8, 2→7, etc.
        new_file = chr(ord('h') - (ord(file) - ord('a')))
        new_rank = chr(ord('8') - (ord(rank) - ord('1')))
        
        transformed = new_file + new_rank
        self.logger.debug(f"Transformed {square} to {transformed} (black perspective)")
        return transformed


class ChessGame:
    """Enhanced chess game state tracking using python-chess library and Stockfish."""
    
    def __init__(self):
        self.board = chess.Board()  # Initialize standard chess board
        self.engine_path = "/home/davidcoyne/Project/shared_folder/chess-robot-integration/Stockfish/src/stockfish"
        self.engine = None
        self._init_engine()
        
    def _init_engine(self):
        """Initialize Stockfish engine."""
        try:
            if Path(self.engine_path).exists():
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                print("Stockfish engine initialized successfully")
            else:
                print(f"Warning: Stockfish not found at {self.engine_path}")
                print("Engine analysis will not be available")
        except Exception as e:
            print(f"Error initializing Stockfish: {str(e)}")
            print("Engine analysis will not be available")
    
    def parse_move(self, move_str: str) -> Optional[Tuple[str, str]]:
        """Parse chess move and return source and target squares.
        
        This enhanced parser handles:
        - Standard piece moves (e.g., Nf3, Bb5)
        - Pawn moves (e.g., e4, d5)
        - Captures with pieces (e.g., Bxc6, Nxd4)
        - Pawn captures (e.g., exd5, bxc3)
        - Castling (O-O, O-O-O)
        - Disambiguated moves (e.g., Nbd2, R1a3)
        - UCI format (e.g., e2e4, b5c6)
        - Explicit pawn notation (e.g., b7-b5)
        """
        try:
            # Normalize input: strip whitespace and check for empty input
            if not move_str or not move_str.strip():
                print("Empty move string provided")
                return self._show_legal_moves()
                
            # Strip any check or checkmate indicators and trim whitespace
            move_str = move_str.replace('+', '').replace('#', '').strip()
            
            # Preserve the original move string for error reporting
            original_move = move_str
            
            # ----- APPROACH 1: Try direct SAN parsing -----
            try:
                # Handle the special case for pawn moves like "e4" or "b5" directly
                if (len(move_str) == 2 and 
                    move_str[0] in "abcdefgh" and 
                    move_str[1] in "12345678"):
                    
                    # This looks like a pawn move to a specific square
                    # Look for a pawn that can move to this square
                    target_square = move_str
                    file_target = move_str[0]
                    
                    # Create a dedicated method to find pawn moves
                    source_square = self._find_pawn_source_for_target(file_target, target_square)
                    if source_square:
                        print(f"Successfully parsed simple pawn move {original_move} as {source_square}->{target_square}")
                        return (source_square, target_square)
                
                # Ensure proper capitalization for pieces to help the parser
                if move_str and move_str[0].lower() in ['n', 'b', 'r', 'q', 'k']:
                    # For piece moves, capitalize the first letter
                    move_str = move_str[0].upper() + move_str[1:]
                
                # Handle castling special case
                if move_str.upper() in ['O-O', 'O-O-O', '0-0', '0-0-0']:
                    # Replace zeros with letters for the chess library
                    move_str = move_str.replace('0', 'O')
                
                # Try SAN notation first
                move = self.board.parse_san(move_str)
                
                # If we get here, move was successfully parsed
                source = chess.square_name(move.from_square)
                target = chess.square_name(move.to_square)
                
                # Log success
                print(f"Successfully parsed move {original_move} as {source}->{target}")
                
                return (source, target)
            except ValueError:
                # SAN parsing failed, continue with other approaches
                pass
                
            # ----- APPROACH 2: Try UCI format (e.g., "e2e4", "b5c6") -----
            if len(move_str) == 4 and move_str[0:2] in chess.SQUARE_NAMES and move_str[2:4] in chess.SQUARE_NAMES:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        source = chess.square_name(move.from_square)
                        target = chess.square_name(move.to_square)
                        print(f"Successfully parsed UCI move {original_move} as {source}->{target}")
                        return (source, target)
                except ValueError:
                    # UCI parsing failed, continue
                    pass
            
            # ----- APPROACH 3: Try explicit source-target with delimiter (e.g., "b7-b5") -----
            if '-' in move_str:
                parts = move_str.split('-')
                if len(parts) == 2 and parts[0] in chess.SQUARE_NAMES and parts[1] in chess.SQUARE_NAMES:
                    try:
                        move = chess.Move.from_uci(parts[0] + parts[1])
                        if move in self.board.legal_moves:
                            source = chess.square_name(move.from_square)
                            target = chess.square_name(move.to_square)
                            print(f"Successfully parsed explicit move {original_move} as {source}->{target}")
                            return (source, target)
                    except ValueError:
                        # Invalid move, continue
                        pass
                
            # ----- APPROACH 4: Special case for piece captures (e.g., "Bxc6") -----
            if len(move_str) >= 4 and move_str[0].isupper() and move_str[1] == 'x':
                piece_symbol = move_str[0]  # The capturing piece (B, N, R, Q, K)
                target_square = move_str[2:]  # The target square
                
                # Find the source square by checking all legal moves
                for move in self.board.legal_moves:
                    source_square = chess.square_name(move.from_square)
                    target = chess.square_name(move.to_square)
                    
                    # Check if this move matches our criteria
                    if target == target_square:
                        piece = self.board.piece_at(move.from_square)
                        if piece and piece.symbol().upper() == piece_symbol:
                            print(f"Successfully parsed piece capture {original_move} as {source_square}->{target_square}")
                            return (source_square, target_square)
                
                # Don't print error message here; continue to other approaches
                
            # ----- APPROACH 5: Handle pawn captures (e.g., "exd5", "bxc3") -----
            if len(move_str) >= 3 and move_str[0].islower() and move_str[1] == 'x':
                # This is a pawn capture notation (e.g., exd5)
                file_from = move_str[0]  # Source file (a-h)
                target_square = move_str[2:]  # Target square
                
                # Find all legal moves that could match this pattern
                for move in self.board.legal_moves:
                    source_square = chess.square_name(move.from_square)
                    target = chess.square_name(move.to_square)
                    
                    # Check if this is a pawn move to the target square from the correct file
                    if target == target_square and source_square[0] == file_from:
                        piece = self.board.piece_at(move.from_square)
                        if piece and piece.piece_type == chess.PAWN:
                            print(f"Successfully parsed pawn capture {original_move} as {source_square}->{target_square}")
                            return (source_square, target_square)
                
                # Don't print error message here; continue to other approaches
            
            # ----- APPROACH 6: Handle disambiguation (e.g., "Nbd2", "R1a3") -----
            if len(move_str) >= 4 and move_str[0].isupper():
                piece_symbol = move_str[0]
                disambiguation = move_str[1]
                target_square = move_str[2:]
                
                # Check if disambiguation is a file (a-h) or rank (1-8)
                is_file = disambiguation.islower()
                is_rank = disambiguation.isdigit()
                
                if is_file or is_rank:
                    for move in self.board.legal_moves:
                        source_square = chess.square_name(move.from_square)
                        target = chess.square_name(move.to_square)
                        
                        if target == target_square:
                            piece = self.board.piece_at(move.from_square)
                            if piece and piece.symbol().upper() == piece_symbol:
                                # Check if disambiguation matches
                                if (is_file and source_square[0] == disambiguation) or \
                                   (is_rank and source_square[1] == disambiguation):
                                    print(f"Successfully parsed disambiguated move {original_move} as {source_square}->{target_square}")
                                    return (source_square, target_square)
            
            # ----- APPROACH 7: Handle piece moves like "Nf3" -----
            if len(move_str) >= 2 and move_str[0].isupper() and move_str[0] in "NBRQK":
                piece_symbol = move_str[0]
                target_square = move_str[1:]
                
                # Track all matches for potentially ambiguous moves
                matches = []
                
                for move in self.board.legal_moves:
                    target = chess.square_name(move.to_square)
                    if target == target_square:
                        piece = self.board.piece_at(move.from_square)
                        if piece and piece.symbol().upper() == piece_symbol:
                            source_square = chess.square_name(move.from_square)
                            matches.append((source_square, target_square))
                            
                # If exactly one match, return it
                if len(matches) == 1:
                    source_square, target_square = matches[0]
                    print(f"Successfully parsed piece move {original_move} as {source_square}->{target_square}")
                    return matches[0]
                elif len(matches) > 1:
                    # Ambiguous - prefer match from the closest rank
                    # Add better disambiguation logic here if needed
                    print(f"Ambiguous piece move {original_move}. Possible interpretations: {matches}")
                    return matches[0]  # Return first match as fallback
            
            # ----- APPROACH 8: ENHANCED - Try explicit source square for pawn move -----
            # Handle notations like "b2b4" or "b7-b5" specifically for pawns
            if len(move_str) == 4:
                source_square = move_str[0:2]
                target_square = move_str[2:4]
                
                if (source_square in chess.SQUARE_NAMES and 
                    target_square in chess.SQUARE_NAMES):
                    
                    # Check if this is a valid pawn move
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in self.board.legal_moves:
                            # Check if it's actually a pawn
                            piece = self.board.piece_at(chess.parse_square(source_square))
                            if piece and piece.piece_type == chess.PAWN:
                                print(f"Successfully parsed explicit pawn move {original_move} as {source_square}->{target_square}")
                                return (source_square, target_square)
                    except ValueError:
                        pass
            
            # ----- APPROACH 9: Second attempt at simple pawn moves ("e4", "b5") -----
            # This is a dedicated test for pawn moves using a simple algorithm
            if len(move_str) == 2 and move_str[0] in "abcdefgh" and move_str[1] in "12345678":
                file_char = move_str[0]  # File (a-h)
                rank_char = move_str[1]  # Rank (1-8)
                target_square = move_str  # The full square name (e.g., "e4")
                
                # For white pawns moving up, check one and two squares behind
                if self.board.turn:  # White's turn
                    potential_sources = [
                        file_char + str(int(rank_char) - 1),  # One square behind
                        file_char + str(int(rank_char) - 2)   # Two squares behind (for first move)
                    ]
                else:  # Black's turn
                    potential_sources = [
                        file_char + str(int(rank_char) + 1),  # One square behind
                        file_char + str(int(rank_char) + 2)   # Two squares behind (for first move)
                    ]
                
                # Check if any of these moves are legal
                for source in potential_sources:
                    if source in chess.SQUARE_NAMES:  # Valid square
                        try:
                            move = chess.Move.from_uci(source + target_square)
                            if move in self.board.legal_moves:
                                # Check if it's actually a pawn
                                piece = self.board.piece_at(chess.parse_square(source))
                                if piece and piece.piece_type == chess.PAWN:
                                    print(f"Successfully parsed pawn move {original_move} as {source}->{target_square}")
                                    return (source, target_square)
                        except ValueError:
                            continue
            
            # ----- APPROACH 10: Match any legal move with the same target square -----
            # This is a fallback approach for when standard notation fails
            if len(move_str) == 2 and move_str in chess.SQUARE_NAMES:
                target_square = move_str
                possible_moves = []
                
                for move in self.board.legal_moves:
                    if chess.square_name(move.to_square) == target_square:
                        source_square = chess.square_name(move.from_square)
                        piece = self.board.piece_at(move.from_square)
                        piece_type = piece.piece_type if piece else None
                        possible_moves.append((source_square, target_square, piece_type))
                
                # If we found exactly one possible move, return it
                if len(possible_moves) == 1:
                    source_square, target_square, _ = possible_moves[0]
                    print(f"Found unique move to {original_move}: {source_square}->{target_square}")
                    return (source_square, target_square)
                    
                # If we found multiple possible moves, prioritize pawns
                elif len(possible_moves) > 1:
                    # Look for pawn moves first as they are most common
                    pawn_moves = [(src, tgt) for src, tgt, piece_type in possible_moves 
                                 if piece_type == chess.PAWN]
                    
                    if len(pawn_moves) == 1:
                        source_square, target_square = pawn_moves[0]
                        print(f"Guessing pawn move to {original_move}: {source_square}->{target_square}")
                        return (source_square, target_square)
                        
                    # If multiple pawn moves or no pawn moves, take the first potential move
                    if possible_moves:
                        source_square, target_square, _ = possible_moves[0]
                        print(f"Ambiguous move to square {original_move}. Using {source_square}->{target_square}")
                        return (source_square, target_square)
            
            # ----- FINAL RESORT: No successful parsing -----
            return self._show_legal_moves(original_move)

        except Exception as e:
            # Log the error but don't let it propagate
            print(f"Error parsing move {move_str}: {str(e)}")
            
            # For debugging purposes only
            import traceback
            traceback.print_exc()
            
            # Show legal moves even after an error
            return self._show_legal_moves()

    def _show_legal_moves(self, attempted_move=None):
        """Helper method to display legal moves and return None."""
        if attempted_move:
            print(f"Could not parse move: {attempted_move}")
        
        try:
            # Group moves by type for better readability
            legal_moves = [self.board.san(move) for move in self.board.legal_moves]
            piece_moves = []
            pawn_moves = []
            captures = []
            specials = []
            
            for move in legal_moves:
                if 'x' in move:
                    captures.append(move)
                elif move in ['O-O', 'O-O-O']:
                    specials.append(move)
                elif move[0].isupper():
                    piece_moves.append(move)
                else:
                    pawn_moves.append(move)
                    
            # Print in organized groups
            if pawn_moves:
                print(f"Pawn moves: {', '.join(pawn_moves)}")
            if piece_moves:
                print(f"Piece moves: {', '.join(piece_moves)}")
            if captures:
                print(f"Captures: {', '.join(captures)}")
            if specials:
                print(f"Special moves: {', '.join(specials)}")
                
            # Print all together for compatibility
            print(f"Legal moves: {', '.join(legal_moves)}")
        except Exception as e:
            print(f"Error showing legal moves: {str(e)}")
        
        return None

    def _find_pawn_source_for_target(self, file_target: str, target_square: str) -> Optional[str]:
        """
        Helper method to find the source square for a pawn that can legally move to the target square.
        
        Args:
            file_target: The file (a-h) of the target square
            target_square: The full target square (e.g., 'e4')
        
        Returns:
            The source square if found, None otherwise
        """
        if not (file_target in "abcdefgh" and target_square[1] in "12345678"):
            return None
            
        rank_target = int(target_square[1])
        
        # Determine possible source squares based on color and target rank
        possible_sources = []
        
        # White pawns move up the ranks (1->2->...)
        if self.board.turn == chess.WHITE:
            # Normal one-square move
            if 2 <= rank_target <= 8:
                possible_sources.append(file_target + str(rank_target - 1))
            # Two-square first move from rank 2 to 4
            if rank_target == 4:
                possible_sources.append(file_target + '2')
        # Black pawns move down the ranks (8->7->...)
        else:
            # Normal one-square move
            if 1 <= rank_target <= 7:
                possible_sources.append(file_target + str(rank_target + 1))
            # Two-square first move from rank 7 to 5
            if rank_target == 5:
                possible_sources.append(file_target + '7')
        
        # Check each possible source square
        for source in possible_sources:
            try:
                # Create a move and check if it's legal
                move = chess.Move.from_uci(source + target_square)
                if move in self.board.legal_moves:
                    piece = self.board.piece_at(chess.parse_square(source))
                    if piece and piece.piece_type == chess.PAWN:
                        return source
            except ValueError:
                continue
        
        # Expand search to include captures
        # White pawn captures move diagonally up
        if self.board.turn == chess.WHITE and 2 <= rank_target <= 8:
            # Check the file to the left (if not on a-file)
            if file_target > 'a':
                left_source = chr(ord(file_target) - 1) + str(rank_target - 1)
                try:
                    move = chess.Move.from_uci(left_source + target_square)
                    if move in self.board.legal_moves:
                        return left_source
                except ValueError:
                    pass
            # Check the file to the right (if not on h-file)
            if file_target < 'h':
                right_source = chr(ord(file_target) + 1) + str(rank_target - 1)
                try:
                    move = chess.Move.from_uci(right_source + target_square)
                    if move in self.board.legal_moves:
                        return right_source
                except ValueError:
                    pass
        
        # Black pawn captures move diagonally down
        elif self.board.turn == chess.BLACK and 1 <= rank_target <= 7:
            # Check the file to the left (if not on a-file)
            if file_target > 'a':
                left_source = chr(ord(file_target) - 1) + str(rank_target + 1)
                try:
                    move = chess.Move.from_uci(left_source + target_square)
                    if move in self.board.legal_moves:
                        return left_source
                except ValueError:
                    pass
            # Check the file to the right (if not on h-file)
            if file_target < 'h':
                right_source = chr(ord(file_target) + 1) + str(rank_target + 1)
                try:
                    move = chess.Move.from_uci(right_source + target_square)
                    if move in self.board.legal_moves:
                        return right_source
                except ValueError:
                    pass
        
        # No valid pawn move found
        return None
    
    def get_move_info(self, move_str: str) -> Optional[dict]:
        """Get detailed move information including piece type, capture status, etc."""
        try:
            # Try to parse the move
            squares = self.parse_move(move_str)
            if not squares:
                return None
                
            source, target = squares
            
            # Get piece information
            piece = self.board.piece_at(chess.parse_square(source))
            if not piece:
                return None
                
            # Build move info
            move_info = {
                'piece_type': PieceType[piece.piece_type_name.upper()],
                'source_square': source,
                'target_square': target,
                'capture': self.board.piece_at(chess.parse_square(target)) is not None,
                'color': Color.WHITE if piece.color else Color.BLACK
            }
            
            return move_info
            
        except Exception as e:
            print(f"Error getting move info: {str(e)}")
            return None
    
    def convert_to_uci(self, san_move: str) -> str:
        """Convert SAN move (e.g., 'Nc3') to UCI format (e.g., 'b1c3')."""
        try:
            move = self.board.parse_san(san_move)
            return move.uci()
        except ValueError:
            raise ValueError(f"Invalid move: {san_move}")
    
    def get_evaluation(self) -> Optional[Tuple[float, str]]:
        """Get position evaluation from Stockfish with mate score handling."""
        try:
            if not self.engine:
                return None
                
            # Use stored configuration or defaults
            config = getattr(self, 'engine_config', {
                "time_limit": 0.1,
                "depth_limit": None
            })
                
            # Get engine analysis with configured limits
            info = self.engine.analyse(
                self.board, 
                chess.engine.Limit(
                    time=config.get("time_limit", 0.1),
                    depth=config.get("depth_limit")
                )
            )
            
            # Extract score information
            score_obj = info["score"].relative
            
            # Check if it's a mate score
            if score_obj.is_mate():
                # Handle mate scores (-M3, M4, etc.)
                mate_in = score_obj.mate()
                
                # Convert mate score to a large value with sign
                # M1 = +/- 10000, M2 = +/- 9999, etc.
                score_float = 10000.0 if mate_in > 0 else -10000.0
                if mate_in != 0:  # Avoid division by zero
                    score_float = score_float * (1.0 - (abs(mate_in) - 1) / 1000.0)
                    
                # Get best move
                best_move = self.board.san(info["pv"][0])
                
                # Include mate information in return value
                return (score_float, best_move, mate_in)
            
            # Regular score (in centipawns)
            score = score_obj.score()
            
            # Convert score to float (in pawns)
            if score is not None:
                score_float = score / 100.0  # Convert centipawns to pawns
                
                # Get best move
                best_move = self.board.san(info["pv"][0])
                
                return (score_float, best_move, None)  # No mate
            return None
                
        except Exception as e:
            logger.error(f"Error getting evaluation: {e}")
            logger.debug(traceback.format_exc())
            return None
            
    def display_state(self, flipped=False) -> None:
        """Display current game state with evaluation."""
        # Get current position evaluation
        eval_info = self.get_evaluation()
        
        # Print board state with orientation based on flipped parameter
        print("\nCurrent Board Position:")
        
        if not flipped:
            # Standard orientation (White perspective)
            print("  a b c d e f g h")
            print(" ┌─────────────────┐")
            
            for rank in range(7, -1, -1):
                print(f"{rank + 1}│", end=" ")
                for file in range(8):
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    if piece is None:
                        print(".", end=" ")
                    else:
                        # Use Unicode chess pieces
                        piece_symbols = {
                            'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
                            'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'
                        }
                        print(piece_symbols.get(piece.symbol(), piece.symbol()), end=" ")
                print("│")
            
            print(" └─────────────────┘")
            print("  a b c d e f g h")
        else:
            # Flipped orientation (Black perspective)
            print("  h g f e d c b a")
            print(" ┌─────────────────┐")
            
            for rank in range(0, 8):
                print(f"{rank + 1}│", end=" ")
                for file in range(7, -1, -1):
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    if piece is None:
                        print(".", end=" ")
                    else:
                        # Use Unicode chess pieces
                        piece_symbols = {
                            'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
                            'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'
                        }
                        print(piece_symbols.get(piece.symbol(), piece.symbol()), end=" ")
                print("│")
            
            print(" └─────────────────┘")
            print("  h g f e d c b a")
        
        # Print evaluation if available
        if eval_info:
            # Check if the evaluation includes mate information
            if len(eval_info) >= 3:
                score, best_move, mate_in = eval_info
            else:
                score, best_move = eval_info
                mate_in = None
                
            turn = "White" if self.board.turn else "Black"
            print(f"\nCurrent turn: {turn}")
            
            # Display mate information if available
            if mate_in is not None:
                if mate_in > 0:
                    print(f"Evaluation: Checkmate in {mate_in} moves")
                    print(f"Best move: {best_move}")
                    print(f"{'White' if self.board.turn else 'Black'} can force checkmate")
                else:
                    print(f"Evaluation: Getting checkmated in {abs(mate_in)} moves")
                    print(f"Best move: {best_move} (trying to delay)")
                    print(f"{'Black' if self.board.turn else 'White'} can force checkmate")
            else:
                # Regular evaluation display
                print(f"Evaluation: {score:+.2f}")
                print(f"Best move: {best_move}")
                
                # Print advantage
                if abs(score) > 0.5:
                    advantage = "White" if score > 0 else "Black"
                    print(f"{advantage} has the advantage")
        else:
            print("\nEngine evaluation not available")
        
        # Print move history
        if len(self.board.move_stack) > 0:
            print("\nMove history:")
            moves = [move.uci() for move in self.board.move_stack]
            for i, move in enumerate(moves):
                if i % 2 == 0:
                    print(f"{i//2 + 1}. {move}", end=" ")
                else:
                    print(f"{move}")
            if len(moves) % 2 == 1:
                print()  # New line if odd number of moves
            
    def is_capture(self, square: str) -> bool:
        """Check if a square has a piece that can be captured."""
        sq = chess.parse_square(square)
        return self.board.piece_at(sq) is not None
    
    def get_piece_on_square(self, square: str) -> Optional[ChessPiece]:
        """Get piece information for a square."""
        if not square:
            return None
            
        sq = chess.parse_square(square)
        piece = self.board.piece_at(sq)
        
        if not piece:
            return None
            
        # Convert chess.Piece to our ChessPiece type
        piece_type = {
            chess.PAWN: PieceType.PAWN,
            chess.KNIGHT: PieceType.KNIGHT,
            chess.BISHOP: PieceType.BISHOP,
            chess.ROOK: PieceType.ROOK,
            chess.QUEEN: PieceType.QUEEN,
            chess.KING: PieceType.KING
        }[piece.piece_type]
        
        color = Color.WHITE if piece.color == chess.WHITE else Color.BLACK
        
        return ChessPiece(
            type=piece_type,
            color=color,
            square=square
        )
    
    def update_position(self, from_square: str, to_square: str):
        """Update board state after a move."""
        move = chess.Move.from_uci(f"{from_square}{to_square}")
        self.board.push(move)  # Updates internal board state
        # Display new state after move
        self.display_state()
        
    def get_turn(self) -> Color:
        """Get current player's turn."""
        return Color.WHITE if self.board.turn else Color.BLACK
    
    def configure_strength(self, skill_level=20, depth_limit=None, time_limit=0.1):
        """
        Configure Stockfish engine strength parameters.
        
        Args:
            skill_level: Integer from 0-20, with 20 being strongest (default: 20)
            depth_limit: Maximum search depth (default: None, meaning no limit)
            time_limit: Time to think in seconds (default: 0.1)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.engine:
                print("Stockfish engine not available!")
                return False
                
            # Set skill level (0-20)
            self.engine.configure({"Skill Level": skill_level})
            
            # Store configuration for future reference
            self.engine_config = {
                "skill_level": skill_level,
                "depth_limit": depth_limit,
                "time_limit": time_limit
            }
            
            print(f"Engine strength configured: skill={skill_level}, depth={depth_limit}, time={time_limit}s")
            return True
            
        except Exception as e:
            print(f"Error configuring engine strength: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.engine:
            self.engine.quit()


class PrinterController:
    """Controls 3D printer movement and gripper operations with adaptive batching."""
    MAX_BATCH_SIZE = 1024
    
    # Global movement constants for consistent operation
    SAFE_HEIGHT = 70.0        # Safe travel height for all movements
    ACCESS_Y = 220.0          # Y position for board access
    GRIPPER_OFFSET = 100.0    # Offset between nozzle and gripper


    # Core printer control methods
    def __init__(self, config: PrinterConfig, chess_config: ChessBoardConfig, 
                 chess_piece_config: ChessPieceConfig, gripper_config: GripperConfig, 
                 storage_config: StorageConfig):
        print("Starting PrinterController initialization...")
        self.is_calibrating = False    
        self.config = config
        print("Config loaded")
        self.z_offset = None
        self.gripper_state = "closed"
        print("Setting up logging...")
        self._setup_logging()
        print("Logging setup complete")
        print("Loading Z offset...")
        self._load_z_offset()
        print("Z offset loaded")
        print("Initializing chess game state...")
        self.chess_game = ChessGame()
        print("Chess game state initialized")
        self.chess_config = chess_config
        self.chess_mapper = ChessPositionMapper(chess_config)
        print("Chess board mapping initialised")
        self.chess_piece_config = chess_piece_config
        self.piece_mapper = ChessPieceMapper(chess_piece_config)
        print("Chess piece settings initialised")
        self.gripper_config = gripper_config
        self.gripper_settings = GripperSettingsMapper(gripper_config)
        print("Gripper settings initialised")
        self.storage_config = storage_config
        self.storage_mapper = StoragePositionMapper(storage_config)
        print("Storage positions initialised")
        self.current_turn = Color.WHITE
        print("Move tracking initialized")
        # Initialize adaptive batching buffer:
        self.command_buffer = []
        self.buffer_size = 0
        
        try:
            print("Initialising GPIO...")
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise ConnectionError("Failed to connect to pigpio daemon")
            print("GPIO initialised")
            print(f"Opening serial port {self.config.printer_port}...")
            self.ser = serial.Serial(
                self.config.printer_port,
                self.config.baud_rate,
                timeout=2
            )
            time.sleep(0.2)  # Allow serial to stabilize
            print("Serial port opened successfully")
            print("Opening gripper to starting position...")
            self.pi.set_servo_pulsewidth(self.config.servo_pin, self.config.gripper_open_pw)
            time.sleep(0.005)
            self.gripper_state = "open"
            print("Gripper initialized to open position")
        except (serial.SerialException, ConnectionError) as e:
            print(f"Initialisation error: {str(e)}")
            raise
        print("PrinterController initialisation completed successfully")

    def _setup_logging(self) -> None:
        self.logger = logging.getLogger("PrinterController")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def send_gcode(self, command: str) -> List[str]:
        """Optimized G-code sending with selective response waiting."""
        try:
            self.ser.write(f"{command}\n".encode())
            # Only wait for response on position queries and movement completion
            if command.startswith(('M114', 'M400', 'G28')):
                time.sleep(0.001)  # Minimal delay for response
                return [line.decode().strip() for line in self.ser.readlines()]
            return []
        except serial.SerialException as e:
            self.logger.error(f"Serial communication error: {str(e)}")
            raise

    def get_position(self) -> Optional[Dict[str, float]]:
            try:
                response = self.send_gcode("M114")
                for line in response:
                    if "X:" in line and "Y:" in line and "Z:" in line:
                        coords = self.parse_position(line)
                        return coords
                self.logger.warning("Could not get position in mm")
                return None
            except Exception as e:
                self.logger.error(f"Error getting position: {str(e)}")
                return None

    def parse_position(self, position_string: str) -> Dict[str, float]:
        coords = {}
        if "Count" in position_string:
            position_string = position_string.split("Count")[0]
        parts = position_string.split()
        for part in parts:
            for axis in ['X:', 'Y:', 'Z:']:
                if part.startswith(axis):
                    coords[axis[0]] = float(part[2:])
        return coords

    def show_status(self):
        try:
            position = self.get_position()
            if position:
                self.logger.info(
                    f"\nCurrent positions:"
                    f"\n  Nozzle:  X={position['X']:.1f}mm, Y={position['Y']:.1f}mm, Z={position['Z']:.1f}mm"
                    f"\n  Gripper: X={position['X']:.1f}mm, Y={position['Y']:.1f}mm, Z={position['Z']-self.GRIPPER_OFFSET:.1f}mm"
                )
        except Exception as e:
            self.logger.error(f"Error showing status: {str(e)}")

    def home_axes(self) -> None:
        try:
            self.logger.info("Homing X and Y axes...")
            self.send_gcode("G28 X Y")
            time.sleep(0.001)
            if self.z_offset is None:
                self._load_z_offset()
            if self.z_offset is None:
                self.logger.info("No Z offset found - starting calibration...")
                self.recalibrate_z_offset()
            else:
                self.logger.info(f"Moving to Z reference position...")
                self.send_gcode(f"G1 Z{self.z_offset} F{self.config.z_feedrate}")
                time.sleep(0.001)
                self._ensure_z_reference()
            position = self.get_position()
            if position:
                self.logger.info(
                    f"\nHoming completed. Final positions:"
                    f"\n  Nozzle:  X={position['X']:.1f}mm, Y={position['Y']:.1f}mm, Z=0.0mm"
                    f"\n  Gripper: X={position['X']:.1f}mm, Y={position['Y']:.1f}mm, Z={-self.GRIPPER_OFFSET:.1f}mm"
                )
        except Exception as e:
            self.logger.error(f"Homing failed: {str(e)}")
            raise

    def test_position_commands(self) -> None:
        try:
            self.logger.info("Testing position commands...")
            self.home_axes()
            self.logger.info("Testing X axis...")
            self.move_nozzle_smooth([('X', 50)])
            time.sleep(1)
            self.move_nozzle_smooth([('X', -50)])
            self.logger.info("Testing Y axis...")
            self.move_nozzle_smooth([('Y', 50)])
            time.sleep(1)
            self.move_nozzle_smooth([('Y', -50)])
            self.logger.info("Testing Z axis...")
            self.move_nozzle_smooth([('Z', 20)])
            time.sleep(1)
            self.move_nozzle_smooth([('Z', -20)])
            self.logger.info("Testing gripper...")
            self.open_gripper()
            time.sleep(1)
            self.close_gripper()
            self.logger.info("Position tests completed")
        except Exception as e:
            self.logger.error(f"Test failed: {str(e)}")

    def add_command_to_buffer(self, command: str) -> None:
        cmd_size = len(command.encode('utf-8')) + 1  # +1 for newline
        if self.buffer_size + cmd_size > self.MAX_BATCH_SIZE:
            self.flush_command_buffer()
        self.command_buffer.append(command)
        self.buffer_size += cmd_size

    def flush_command_buffer(self) -> List[str]:
        """Improved command buffer flushing with proper response waiting."""
        if not self.command_buffer:
            return []
        
        # Check if any movement commands are in the buffer
        has_movement_commands = any(cmd.startswith(('G0', 'G1', 'G2', 'G3')) 
                                  for cmd in self.command_buffer)
        
        # Add M400 if there are movement commands and none exists already
        if has_movement_commands and not any(cmd.startswith('M400') for cmd in self.command_buffer):
            self.command_buffer.append("M400")
        
        # Combine all commands into a single transmission
        batched_command = "\n".join(self.command_buffer) + "\n"
        self.ser.write(batched_command.encode())
        
        # More substantial waiting time for command processing
        wait_time = 0.05  # 50ms minimum
        
        # Extra time if movement commands are present
        if has_movement_commands:
            wait_time = 0.2  # 200ms for movements
        
        # Wait for commands to be processed
        time.sleep(wait_time)
        
        # Read all available responses
        responses = [line.decode().strip() for line in self.ser.readlines()]
        
        # If we expected a response but got none, wait longer and try again
        if (has_movement_commands or any(cmd.startswith(('M114', 'G28')) 
                                       for cmd in self.command_buffer)) and not responses:
            time.sleep(0.3)  # Additional 300ms wait
            responses = [line.decode().strip() for line in self.ser.readlines()]
        
        # Clear the buffer
        self.command_buffer = []
        self.buffer_size = 0
        
        return responses

    def send_batched_commands(self, commands: List[str]) -> List[str]:
        for cmd in commands:
            self.add_command_to_buffer(cmd)
        return self.flush_command_buffer()

    def cleanup(self) -> None:
        try:
            self.chess_game.cleanup()
            self.pi.set_servo_pulsewidth(self.config.servo_pin, 0)
            self.pi.stop()
            self.ser.close()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")


    # Movement and position methods 
    def move_nozzle_smooth(self, movements: List[Tuple[str, float]], feedrate: Optional[int] = None) -> bool:
        """Enhanced multi-axis movement with proper completion waiting."""
        try:
            # [Existing validation code]
            
            # Combine movements into single command
            movement_str = " ".join([f"{axis}{value}" for axis, value in movements])
            commands = [
                "G91",  # Relative positioning
                f"G1 {movement_str} F{feedrate}",
                "G90",  # Back to absolute
                "M400"  # Wait for completion
            ]
            
            # Send the commands
            self.send_batched_commands(commands)
            
            # CRITICAL: Add explicit wait for movement to complete
            self.wait_for_movement_complete()
            
            return True
        except Exception as e:
            self.logger.error(f"Smooth movement error: {str(e)}")
            return False
            
    def move_to_square_smooth(self, square: str, maintain_z: bool = True) -> bool:
        """Optimized square movement with combined commands."""
        try:
            position = self.get_transformed_position(square)
            if not position:
                self.logger.error(f"No calibrated position for {square}")
                return False
                
            current_pos = self.get_position()
            if current_pos:
                dx = position['x'] - current_pos['X']
                dy = position['y'] - current_pos['Y']
                
                if dx != 0 or dy != 0:
                    # Combine all movement commands into single batch
                    commands = [
                        "G91",  # Relative positioning
                        f"G1 X{dx} Y{dy} F{self.config.xy_feedrate}",
                        "G90",  # Back to absolute
                        "M400"  # Wait for movement completion
                    ]
                    self.send_batched_commands(commands)
                    
                self.logger.info(f"Moved to chess position {square}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to move to square {square}: {str(e)}")
            return False

    def move_piece_height(self, height_diff: float) -> bool:
        try:
            return self.move_nozzle_smooth([('Z', height_diff)])
        except Exception as e:
            self.logger.error(f"Error moving to piece height: {str(e)}")
            return False

    def move_to_z_height(self, target_height: float) -> bool:
        try:
            current_pos = self.get_position()
            if current_pos:
                diff = target_height - current_pos['Z']
                return self.move_nozzle_smooth([('Z', diff)])
            return False
        except Exception as e:
            self.logger.error(f"Error moving to Z height: {str(e)}")
            return False

    def move_to_storage(self, location: str) -> bool:
        try:
            position = self.storage_mapper.get_position(location)
            if not position:
                self.logger.error(f"No calibrated position for {location}")
                return False
            current_pos = self.get_position()
            if current_pos:
                dx = position['x'] - current_pos['X']
                dy = position['y'] - current_pos['Y']
                dz = position['z'] - current_pos['Z']
                if dz > 0:
                    self.move_nozzle_smooth([('Z', dz)])
                if dx != 0 or dy != 0:
                    self.send_batched_commands([
                        "G91",
                        f"G1 X{dx} Y{dy} F{self.config.xy_feedrate}",
                        "G90"
                    ])
                if dz < 0:
                    self.move_nozzle_smooth([('Z', dz)])
                self.show_status()
                self.logger.info(f"Moved to storage position {location}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to move to storage {location}: {str(e)}")
            return False

    def _validate_movement(self, axis: str, value: float) -> bool:
        try:
            position = self.get_position()
            if position is None:
                return False
            new_pos = position.copy()
            new_pos[axis] = position[axis] + value
            if not self.is_calibrating and axis in ["X", "Y"]:
                if abs(new_pos[axis]) > self.config.max_xy_distance:
                    self.logger.warning(f"{axis} movement out of safe range: {new_pos[axis]}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Movement validation error: {str(e)}")
            return False

    def wait_for_movement_complete(self):
        """Enhanced movement completion check with proper waiting."""
        try:
            # Send M400 to request movement completion 
            response = self.send_gcode("M400")
            
            # Significant delay needed for mechanical completion
            time.sleep(0.2)  # 200ms - much more realistic
            
            # Look for acknowledgment in response
            acknowledgment_received = any("ok" in line.lower() for line in response)
            
            if not acknowledgment_received:
                # If no acknowledgment received, add extra wait time
                time.sleep(0.5)  # 500ms additional wait
                
            # Double-check position to confirm completion
            pos = self.get_position()
            if pos is None:
                self.logger.warning("Could not verify movement completion")
                # Additional wait if position check failed
                time.sleep(0.5)
                
            return True
        except Exception as e:
            self.logger.error(f"Error waiting for movement completion: {str(e)}")
            # Add safety delay on error
            time.sleep(0.5)
            return False

    def set_board_orientation(self, player_is_white: bool) -> None:
        """Set the board orientation based on player color"""
        self.chess_mapper.flip_board(not player_is_white)
        self.logger.info(f"Board orientation set for player as {'white' if player_is_white else 'black'}")
        
        # You might want to move the board to a specific position
        # to make it easy for the player to interact with it
        if not player_is_white:
            self.move_to_board_access_position()

    def get_transformed_position(self, square: str) -> Optional[Dict[str, float]]:
        """Get position coordinates with proper orientation transformation"""
        transformed_square = self.chess_mapper.transform_square(square)
        return self.chess_mapper.positions.get(transformed_square)

    def display_square(self, square: str) -> str:
        """Display square name in player's perspective"""
        if not hasattr(self.chess_mapper, 'flipped') or not self.chess_mapper.flipped:
            return square.upper()
        
        # For black player, we display the transformed coordinates
        file = square[0]
        rank = square[1]
        new_file = chr(ord('h') - (ord(file) - ord('a')))
        new_rank = chr(ord('8') - (ord(rank) - ord('1')))
        
        return (new_file + new_rank).upper()
    
    def show_board_orientation(self) -> None:
        """Display the current board orientation to help the player"""
        print("\nBoard orientation:")
        if not self.chess_mapper.flipped:
            print("    a b c d e f g h  ← WHITE side (Standard)")
            print("  ┌─────────────────┐")
            print("8 │ . . . . . . . . │ 8")
            print("7 │ . . . . . . . . │ 7")
            print("6 │ . . . . . . . . │ 6")
            print("5 │ . . . . . . . . │ 5")
            print("4 │ . . . . . . . . │ 4")
            print("3 │ . . . . . . . . │ 3")
            print("2 │ . . . . . . . . │ 2")
            print("1 │ . . . . . . . . │ 1")
            print("  └─────────────────┘")
            print("    a b c d e f g h  ← BLACK side")
        else:
            print("    h g f e d c b a  ← BLACK side (Flipped)")
            print("  ┌─────────────────┐")
            print("1 │ . . . . . . . . │ 1")
            print("2 │ . . . . . . . . │ 2")
            print("3 │ . . . . . . . . │ 3")
            print("4 │ . . . . . . . . │ 4")
            print("5 │ . . . . . . . . │ 5")
            print("6 │ . . . . . . . . │ 6")
            print("7 │ . . . . . . . . │ 7")
            print("8 │ . . . . . . . . │ 8")
            print("  └─────────────────┘")
            print("    h g f e d c b a  ← WHITE side")

    
    # Gripper control methods
    def open_gripper(self, slow_mode: bool = False, force: bool = False) -> bool:
        """
        Enhanced gripper opening with better error handling and state verification.
        
        Args:
            slow_mode (bool): Whether to use slower movement for precision
            force (bool): Force operation even if gripper is already open
            
        Returns:
            bool: True if successful, False otherwise
        """
        MAX_RETRIES = 3
        retry_count = 0
        
        # Skip if already open (unless forced)
        if self.gripper_state == "open" and not force:
            self.logger.debug("Gripper already open, skipping open operation")
            return True
        
        # Try opening with retries on failure
        while retry_count < MAX_RETRIES:
            try:
                retry_count += 1
                if retry_count > 1:
                    self.logger.info(f"Opening gripper (retry {retry_count}/{MAX_RETRIES})" + 
                                   (" (slow mode)" if slow_mode else ""))
                else:
                    self.logger.info("Opening gripper" + (" (slow mode)" if slow_mode else ""))
                
                # Calculate step size and delay based on mode
                step = self.config.gripper_step // 2 if slow_mode else self.config.gripper_step
                delay = self.config.gripper_delay * 2 if slow_mode else self.config.gripper_delay
                
                # Get current servo pulse width
                current_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
                if current_pw <= 0:  # Invalid reading
                    current_pw = self.config.gripper_closed_pw  # Default to closed position
                
                # Move from current to open position in small steps
                from_pw = min(current_pw, self.config.gripper_closed_pw)
                to_pw = self.config.gripper_open_pw
                
                # Calculate range (work backward from closed to open)
                step_range = range(from_pw, to_pw - 1, -step)
                
                # Move in small steps for smooth motion
                for pw in step_range:
                    self.pi.set_servo_pulsewidth(self.config.servo_pin, pw)
                    time.sleep(delay)
                
                # Ensure we end exactly at the open position
                self.pi.set_servo_pulsewidth(self.config.servo_pin, to_pw)
                time.sleep(delay * 2)  # Extra delay to ensure we reached position
                
                # Update internal state
                self.gripper_state = "open"
                
                # Add mechanical settling delay
                time.sleep(0.01)
                
                # Verify the gripper is actually open
                actual_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
                if abs(actual_pw - to_pw) > 20:  # Allow small tolerance
                    self.logger.warning(f"Gripper position verification failed: Expected {to_pw}, got {actual_pw}")
                    # Try again if we've not reached max retries
                    if retry_count < MAX_RETRIES:
                        continue
                
                # Success
                return True
                
            except Exception as e:
                self.logger.error(f"Error opening gripper (attempt {retry_count}): {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Retry after a pause
                time.sleep(0.05)
                
                # If last retry failed, log and return failure
                if retry_count >= MAX_RETRIES:
                    self.logger.error(f"Failed to open gripper after {MAX_RETRIES} attempts")
                    return False
        
        # We shouldn't reach here, but return False just in case
        return False

    def close_gripper(self, slow_mode: bool = False, force: bool = False) -> bool:
        """
        Enhanced gripper closing with better error handling and state verification.
        
        Args:
            slow_mode (bool): Whether to use slower movement for precision
            force (bool): Force operation even if gripper is already closed
            
        Returns:
            bool: True if successful, False otherwise
        """
        MAX_RETRIES = 3
        retry_count = 0
        
        # Skip if already closed (unless forced)
        if self.gripper_state == "closed" and not force:
            self.logger.debug("Gripper already closed, skipping close operation")
            return True
        
        # Try closing with retries on failure
        while retry_count < MAX_RETRIES:
            try:
                retry_count += 1
                if retry_count > 1:
                    self.logger.info(f"Closing gripper (retry {retry_count}/{MAX_RETRIES})" + 
                                   (" (slow mode)" if slow_mode else ""))
                else:
                    self.logger.info("Closing gripper" + (" (slow mode)" if slow_mode else ""))
                
                # Calculate step size and delay based on mode
                step = self.config.gripper_step // 2 if slow_mode else self.config.gripper_step
                delay = self.config.gripper_delay * 2 if slow_mode else self.config.gripper_delay
                
                # Get current servo pulse width
                current_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
                if current_pw <= 0:  # Invalid reading
                    current_pw = self.config.gripper_open_pw  # Default to open position
                
                # Move from current to closed position in small steps
                from_pw = max(current_pw, self.config.gripper_open_pw)
                to_pw = self.config.gripper_closed_pw
                
                # Calculate range (work forward from open to closed)
                step_range = range(from_pw, to_pw + 1, step)
                
                # Move in small steps for smooth motion
                for pw in step_range:
                    self.pi.set_servo_pulsewidth(self.config.servo_pin, pw)
                    time.sleep(delay)
                
                # Ensure we end exactly at the closed position
                self.pi.set_servo_pulsewidth(self.config.servo_pin, to_pw)
                time.sleep(delay * 2)  # Extra delay to ensure we reached position
                
                # Update internal state
                self.gripper_state = "closed"
                
                # Add mechanical settling delay
                time.sleep(0.01)
                
                # Verify the gripper is actually closed
                actual_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
                if abs(actual_pw - to_pw) > 20:  # Allow small tolerance
                    self.logger.warning(f"Gripper position verification failed: Expected {to_pw}, got {actual_pw}")
                    # Try again if we've not reached max retries
                    if retry_count < MAX_RETRIES:
                        continue
                
                # Success
                return True
                
            except Exception as e:
                self.logger.error(f"Error closing gripper (attempt {retry_count}): {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Retry after a pause
                time.sleep(0.05)
                
                # If last retry failed, log and return failure
                if retry_count >= MAX_RETRIES:
                    self.logger.error(f"Failed to close gripper after {MAX_RETRIES} attempts")
                    return False
        
        # We shouldn't reach here, but return False just in case
        return False

    def set_gripper_position(self, target_pw: int, slow_mode: bool = False) -> bool:
        """
        Enhanced gripper positioning with better error handling and state verification.
        
        Args:
            target_pw (int): Target pulse width
            slow_mode (bool): Whether to use slower movement for precision
            
        Returns:
            bool: True if successful, False otherwise
        """
        MAX_RETRIES = 3
        retry_count = 0
        
        # Validate target pulse width is in valid range
        if target_pw < self.config.gripper_open_pw or target_pw > self.config.gripper_closed_pw:
            self.logger.error(f"Invalid target pulse width: {target_pw}. Must be between "
                            f"{self.config.gripper_open_pw} and {self.config.gripper_closed_pw}")
            return False
        
        # Try setting position with retries on failure
        while retry_count < MAX_RETRIES:
            try:
                retry_count += 1
                if retry_count > 1:
                    self.logger.info(f"Setting gripper position to {target_pw} (retry {retry_count}/{MAX_RETRIES})" + 
                                   (" (slow mode)" if slow_mode else ""))
                else:
                    self.logger.info(f"Setting gripper position to {target_pw}" + (" (slow mode)" if slow_mode else ""))
                
                # Calculate step size and delay based on mode
                step = self.config.gripper_step // 2 if slow_mode else self.config.gripper_step
                delay = self.config.gripper_delay * 2 if slow_mode else self.config.gripper_delay
                
                # Get current servo pulse width
                current_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
                if current_pw <= 0:  # Invalid reading
                    # Use state to estimate current position
                    current_pw = (self.config.gripper_closed_pw if self.gripper_state == "closed" 
                                else self.config.gripper_open_pw)
                
                # Check if we're already at target (with tolerance)
                if abs(current_pw - target_pw) < 10:
                    self.logger.debug(f"Gripper already at target position {target_pw}, skipping movement")
                    # Still update the state based on position
                    self._update_gripper_state_from_pw(target_pw)
                    return True
                
                # Determine direction of movement
                if target_pw > current_pw:  # Moving toward closed
                    step_range = range(current_pw, target_pw + 1, step)
                else:  # Moving toward open
                    step_range = range(current_pw, target_pw - 1, -step)
                
                # Move in small steps for smooth motion
                for pw in step_range:
                    self.pi.set_servo_pulsewidth(self.config.servo_pin, pw)
                    time.sleep(delay)
                
                # Ensure we end exactly at the target position
                self.pi.set_servo_pulsewidth(self.config.servo_pin, target_pw)
                time.sleep(delay * 2)  # Extra delay to ensure we reached position
                
                # Update internal state based on position
                self._update_gripper_state_from_pw(target_pw)
                
                # Add mechanical settling delay
                time.sleep(0.01)
                
                # Verify the gripper is at target position
                actual_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
                if abs(actual_pw - target_pw) > 20:  # Allow small tolerance
                    self.logger.warning(f"Gripper position verification failed: Expected {target_pw}, got {actual_pw}")
                    # Try again if we've not reached max retries
                    if retry_count < MAX_RETRIES:
                        continue
                
                # Success
                return True
                
            except Exception as e:
                self.logger.error(f"Error setting gripper position (attempt {retry_count}): {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                # Retry after a pause
                time.sleep(0.05)
                
                # If last retry failed, log and return failure
                if retry_count >= MAX_RETRIES:
                    self.logger.error(f"Failed to set gripper position after {MAX_RETRIES} attempts")
                    return False
        
        # We shouldn't reach here, but return False just in case
        return False

    def _update_gripper_state_from_pw(self, pulse_width: int) -> None:
        """
        Helper method to update gripper state based on pulse width value.
        
        Args:
            pulse_width (int): Current pulse width
        """
        # Calculate midpoint between open and closed positions
        midpoint = (self.config.gripper_open_pw + self.config.gripper_closed_pw) / 2
        
        # Update state based on which side of midpoint we're on
        if pulse_width >= midpoint:
            self.gripper_state = "closed"
        else:
            self.gripper_state = "open"
            
    def verify_gripper_state(self, expected_state: str = None) -> bool:
        """
        Enhanced verification of gripper state with automatic correction.
        
        Args:
            expected_state (str): Expected state ("open", "closed", or None)
            
        Returns:
            bool: True if verification passed, False otherwise
        """
        try:
            # If no expected state provided, just log current state
            if expected_state is None:
                self.logger.info(f"Current gripper state: {self.gripper_state}")
                return True
            
            # Get current pulse width to verify physical state
            current_pw = self.pi.get_servo_pulsewidth(self.config.servo_pin)
            
            # Validate state consistency
            if expected_state == "open":
                # Check if state is already correct
                if self.gripper_state == "open":
                    # Verify physical position if we can get a reading
                    if current_pw > 0:
                        # Allow tolerance in pulse width
                        if abs(current_pw - self.config.gripper_open_pw) > 50:
                            self.logger.warning(f"Gripper should be open but pulse width is {current_pw}")
                            # Force correction
                            return self.open_gripper(force=True)
                    return True
                else:
                    # State inconsistency - correct it
                    self.logger.warning(f"Gripper state inconsistency detected! Expected: {expected_state}, Current: {self.gripper_state}")
                    return self.open_gripper(force=True)
                    
            elif expected_state == "closed":
                # Check if state is already correct
                if self.gripper_state == "closed":
                    # Verify physical position if we can get a reading
                    if current_pw > 0:
                        # Allow tolerance in pulse width
                        if abs(current_pw - self.config.gripper_closed_pw) > 50:
                            self.logger.warning(f"Gripper should be closed but pulse width is {current_pw}")
                            # Force correction
                            return self.close_gripper(force=True)
                    return True
                else:
                    # State inconsistency - correct it
                    self.logger.warning(f"Gripper state inconsistency detected! Expected: {expected_state}, Current: {self.gripper_state}")
                    return self.close_gripper(force=True)
            else:
                self.logger.warning(f"Unknown expected state: {expected_state}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during gripper state verification: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
        
    def emergency_release(self) -> bool:
        """
        Emergency gripper release function for error recovery.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.warning("Performing emergency gripper release")
        
        try:
            # Try multiple approaches to ensure release
            for attempt in range(3):
                # Set directly to open position with maximum pulse width
                self.pi.set_servo_pulsewidth(self.config.servo_pin, self.config.gripper_open_pw)
                time.sleep(0.02)  # Longer delay for mechanical action
                
                # Mark state as open
                self.gripper_state = "open"
                
                # Verify it's actually open
                if self.verify_gripper_state("open"):
                    self.logger.info("Emergency release successful")
                    return True
            
            self.logger.error("Emergency release failed after multiple attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during emergency release: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Last resort - try direct pulse width setting
            try:
                self.pi.set_servo_pulsewidth(self.config.servo_pin, self.config.gripper_open_pw)
                self.gripper_state = "open"
            except:
                pass
                
            return False
        
    def control_gripper(self) -> None:
        while True:
            print("\nGripper Control Mode")
            print("1. Move to position (2300-2500)")
            print("2. Open gripper")
            print("3. Close gripper")
            print("4. Show current settings")
            print("5. Update settings")
            print("6. Return to main menu")
            choice = input("Enter choice (1-6): ").strip()
            if choice == "1":
                try:
                    pos = int(input("Enter position (2300-2500): "))
                    if 2300 <= pos <= 2500:
                        self.pi.set_servo_pulsewidth(self.config.servo_pin, pos)
                        self.logger.info(f"Moved gripper to position {pos}")
                    else:
                        self.logger.error("Position must be between 2300 and 2500")
                except ValueError:
                    self.logger.error("Invalid input! Please enter a number")
            elif choice == "2":
                self.open_gripper()
            elif choice == "3":
                self.close_gripper()
            elif choice == "4":
                settings = self.gripper_settings.get_settings()
                print("\nCurrent Gripper Settings:")
                print(f"Open position: {settings['open_pw']}")
                print(f"Closed position: {settings['closed_pw']}")
                print(f"Step size: {settings['step']}")
                print(f"Delay: {settings['delay']}")
            elif choice == "5":
                self.update_gripper_settings()
            elif choice == "6":
                break
            else:
                print("Invalid choice! Please enter 1-6")


    # Chess gameplay methods
    def play_move(self, notation: str) -> bool:
        """Improved chess move execution with enhanced error handling and gripper reliability."""
        try:
            # Parse the move notation to get source and target squares
            if len(notation) >= 3 and notation[0].islower() and notation[1] == 'x':
                # Try to parse pawn capture notation
                squares = self.chess_game.parse_move(notation)
            else:
                try:
                    move_obj = self.chess_game.board.parse_san(notation)
                    if self.chess_game.board.is_castling(move_obj):
                        return self.play_castling_move(notation)
                    squares = self.chess_game.parse_move(notation)
                except Exception as e:
                    self.logger.error(f"Error parsing move {notation}: {str(e)}")
                    squares = None
                    
            if not squares:
                self.logger.error(f"Invalid move: {notation}")
                return False
                
            source_square, target_square = squares
            piece = self.chess_game.get_piece_on_square(source_square)
            
            if not piece:
                self.logger.error(f"No piece found on {source_square}")
                return False
                
            piece_settings = self.piece_mapper.get_piece_settings(piece.type.value)
            if not piece_settings:
                self.logger.error(f"No settings found for {piece.type.value}")
                return False
            
            # CRITICAL SAFETY CHECK: Ensure gripper is open before starting any move
            if not self.verify_gripper_state("open"):
                self.logger.error("Cannot guarantee gripper is open after verification - retrying once")
                # One more try with forced open
                success = self.open_gripper(force=True)
                if not success:
                    self.logger.error("Failed to ensure gripper is open even after retry - aborting move")
                    return False
            
            # Handle capture if needed
            if self.chess_game.is_capture(target_square):
                captured_piece = self.chess_game.get_piece_on_square(target_square)
                if captured_piece:
                    # Determine proper storage box based on piece color and board orientation
                    if self.chess_mapper.flipped:
                        # When board is flipped (playing as black), reverse the box choices
                        storage_box = 'box_1' if captured_piece.color == Color.WHITE else 'box_2'
                    else:
                        # Standard orientation (playing as white)
                        storage_box = 'box_2' if captured_piece.color == Color.WHITE else 'box_1'
                    self.logger.info(f"Capturing {captured_piece.type.value} on {target_square} to {storage_box}")
                    capture_success = self._capture_piece(target_square, storage_box)
                    if not capture_success:
                        self.logger.error("Failed to capture piece - aborting move")
                        return False
                    
                    # Make sure gripper is open again after capture
                    if not self.verify_gripper_state("open"):
                        self.logger.warning("Gripper not open after capture - forcing open")
                        if not self.open_gripper(force=True):
                            self.logger.error("Failed to open gripper after capture - aborting move")
                            return False
            
            # Get source square coordinates
            source_pos = self.get_transformed_position(source_square)
            if not source_pos:
                self.logger.error(f"No position mapping for square {source_square}")
                return False
            
            # Move to safe height first
            self.logger.info(f"Moving to safe height {self.SAFE_HEIGHT}")
            if not self.move_to_z_height(self.SAFE_HEIGHT):
                self.logger.error("Failed to move to safe height")
                return False
            
            # Move to source square
            self.logger.info(f"Moving to source square {source_square}")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position")
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', source_pos['x'] - current_pos['X']), 
                ('Y', source_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to source square position")
                return False
            
            # Verify gripper is still open before descending
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open before pickup - forcing open")
                if not self.open_gripper(force=True):
                    self.logger.error("Failed to ensure gripper is open before pickup - aborting")
                    return False
            
            # Move to piece height with slower feedrate for precision
            self.logger.info(f"Moving down to piece height {piece_settings['height']}")
            if not self.move_nozzle_smooth([('Z', piece_settings['height'] - self.SAFE_HEIGHT)], 
                                         feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to piece height")
                return False
                
            # Grip the piece
            self.logger.info(f"Gripping piece with pulse width {piece_settings['grip_pw']}")
            if not self.set_gripper_position(piece_settings['grip_pw']):
                self.logger.error("Failed to grip piece")
                # Emergency move back up to safe height
                self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])])
                return False
                
            # Extra delay to ensure grip is secure
            time.sleep(0.02)
            
            # Verify gripper is actually gripping
            if not self.verify_gripper_state("closed"):
                self.logger.warning("Gripper not in closed state after gripping - forcing state update")
                self.gripper_state = "closed"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])],
                                         feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after pickup")
                # Try emergency move to safe height
                if not self.move_to_z_height(self.SAFE_HEIGHT):
                    self.logger.error("Emergency height recovery failed - trying to release piece anyway")
                    self.emergency_release()
                    return False
                
            # Get target square coordinates
            target_pos = self.get_transformed_position(target_square)
            if not target_pos:
                self.logger.error(f"No position mapping for target square {target_square}")
                # Release piece since we can't place it properly
                self.emergency_release()
                return False
                
            # Move to target square
            self.logger.info(f"Moving to target square {target_square}")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position before target move")
                self.emergency_release()
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', target_pos['x'] - current_pos['X']), 
                ('Y', target_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to target square position")
                self.emergency_release()
                return False
            
            # Move down to piece placement height
            self.logger.info("Moving down to place piece")
            if not self.move_nozzle_smooth([('Z', piece_settings['height'] - self.SAFE_HEIGHT)],
                                         feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to piece placement height")
                # Try to get to a reasonable height
                self.move_to_z_height(target_pos['z'] + 20)  # Slightly above target
                self.emergency_release()
                return False
            
            # Release the piece using slow mode for controlled release
            self.logger.info("Releasing piece")
            if not self.open_gripper(slow_mode=True):
                self.logger.error("Failed to release piece normally - attempting emergency release")
                self.emergency_release()
                # Continue with move regardless
            
            # Extra delay to ensure complete release
            time.sleep(0.02)
            
            # Verify gripper is actually open
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open after release - forcing state update")
                self.gripper_state = "open"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height after placing piece")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])],
                                         feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after release")
                # Try emergency move to safe height
                self.move_to_z_height(self.SAFE_HEIGHT)
                # Continue with move regardless
            
            # Update game state
            try:
                move_obj = self.chess_game.board.parse_san(notation)
            except ValueError:
                # For moves that can't be parsed in SAN (like UCI moves)
                move = chess.Move.from_uci(f"{source_square}{target_square}")
                self.chess_game.board.push(move)
            else:
                self.chess_game.board.push(move_obj)
                
            self.current_turn = self.chess_game.get_turn()
            self.show_status()
            
            self.logger.info(f"Successfully executed move: {notation}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing move {notation}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Emergency release in case we're holding a piece
            try:
                self.emergency_release()
            except Exception as release_err:
                self.logger.error(f"Error during emergency release: {str(release_err)}")
            
            # Try to get back to safe height
            try:
                self.move_to_z_height(self.SAFE_HEIGHT)
            except:
                pass
                
            return False
         
    def play_castling_move(self, notation: str) -> bool:
        """Enhanced castling implementation with better error handling, debugging, and transformation awareness."""
        try:
            self.logger.info(f"===== STARTING CASTLING OPERATION: {notation} =====")
            self.logger.info(f"Board orientation: {'BLACK perspective (flipped)' if self.chess_mapper.flipped else 'WHITE perspective (standard)'}")
            
            # Validate this is actually a castling move
            move_obj = self.chess_game.board.parse_san(notation)
            if not self.chess_game.board.is_castling(move_obj):
                self.logger.error("The move is not a castling move!")
                return False
            
            castling_type = "kingside" if self.chess_game.board.is_kingside_castling(move_obj) else "queenside"
            self.logger.info(f"Castling type: {castling_type}")
            
            # Determine king source and target squares (these are already in internal coordinates)
            king_source = chess.square_name(move_obj.from_square)
            king_target = chess.square_name(move_obj.to_square)
            self.logger.info(f"Internal coordinates - King: {king_source} → {king_target}")
            
            # Log the display coordinates (what the user sees)
            display_king_source = self.display_square(king_source)
            display_king_target = self.display_square(king_target)
            self.logger.info(f"Display coordinates - King: {display_king_source} → {display_king_target}")
            
            # Get king piece information
            king_piece = self.chess_game.get_piece_on_square(king_source)
            if not king_piece:
                self.logger.error(f"No king found on {king_source}")
                return False
            
            king_color = "WHITE" if king_piece.color == Color.WHITE else "BLACK"
            self.logger.info(f"King color: {king_color}")
            
            # Get king settings
            king_settings = self.piece_mapper.get_piece_settings(king_piece.type.value)
            if not king_settings:
                self.logger.error("No calibration settings found for the king!")
                return False
            
            # Determine rook squares based on castling type, king color, and board orientation
            # This builds a comprehensive map for all possible scenarios
            # Format: (castling_type, king_color, is_flipped) -> (rook_source, rook_target)
            castling_coords = {
                # Kingside castling
                ("kingside", "WHITE", False): ("h1", "f1"),  # White kingside, standard view
                ("kingside", "WHITE", True): ("h1", "f1"),   # White kingside, flipped view
                ("kingside", "BLACK", False): ("h8", "f8"),  # Black kingside, standard view
                ("kingside", "BLACK", True): ("h8", "f8"),   # Black kingside, flipped view
                
                # Queenside castling
                ("queenside", "WHITE", False): ("a1", "d1"), # White queenside, standard view
                ("queenside", "WHITE", True): ("a1", "d1"),  # White queenside, flipped view
                ("queenside", "BLACK", False): ("a8", "d8"), # Black queenside, standard view
                ("queenside", "BLACK", True): ("a8", "d8"),  # Black queenside, flipped view
            }
            
            # Look up the proper rook coordinates
            key = (castling_type, king_color, self.chess_mapper.flipped)
            if key not in castling_coords:
                self.logger.error(f"No castling coordinates defined for scenario: {key}")
                return False
            
            rook_source, rook_target = castling_coords[key]
            self.logger.info(f"Internal coordinates - Rook: {rook_source} → {rook_target}")
            
            # Log the display coordinates (what the user sees)
            display_rook_source = self.display_square(rook_source)
            display_rook_target = self.display_square(rook_target)
            self.logger.info(f"Display coordinates - Rook: {display_rook_source} → {display_rook_target}")
            
            # Log transformed coordinates that will be used for actual movements
            king_source_pos = self.get_transformed_position(king_source)
            king_target_pos = self.get_transformed_position(king_target)
            rook_source_pos = self.get_transformed_position(rook_source)
            rook_target_pos = self.get_transformed_position(rook_target)
            
            self.logger.info(f"Physical king source position: {king_source_pos}")
            self.logger.info(f"Physical king target position: {king_target_pos}")
            self.logger.info(f"Physical rook source position: {rook_source_pos}")
            self.logger.info(f"Physical rook target position: {rook_target_pos}")
            
            # Validate we have all necessary positions
            if not king_source_pos or not king_target_pos or not rook_source_pos or not rook_target_pos:
                missing = []
                if not king_source_pos: missing.append(f"king_source ({king_source})")
                if not king_target_pos: missing.append(f"king_target ({king_target})")
                if not rook_source_pos: missing.append(f"rook_source ({rook_source})")
                if not rook_target_pos: missing.append(f"rook_target ({rook_target})")
                self.logger.error(f"Missing position mappings for: {', '.join(missing)}")
                return False
                
            # CRITICAL SAFETY CHECK: Ensure gripper is open before starting any move
            if not self.verify_gripper_state("open"):
                self.logger.error("Cannot guarantee gripper is open after verification - retrying once")
                # One more try with forced open
                success = self.open_gripper(force=True)
                if not success:
                    self.logger.error("Failed to ensure gripper is open even after retry - aborting move")
                    return False
            
            # Move to safe height first
            self.logger.info(f"Moving to safe height {self.SAFE_HEIGHT}")
            if not self.move_to_z_height(self.SAFE_HEIGHT):
                self.logger.error("Failed to move to safe height")
                return False
            
            ########################## KING MOVEMENT ##########################
            self.logger.info("===== KING MOVEMENT PHASE =====")
            
            # Move to king source square
            self.logger.info(f"Moving to king source square {king_source} (display: {display_king_source})")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position")
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', king_source_pos['x'] - current_pos['X']), 
                ('Y', king_source_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to king source position")
                return False
            
            # Verify gripper is still open before descending
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open before pickup - forcing open")
                if not self.open_gripper(force=True):
                    self.logger.error("Failed to ensure gripper is open before pickup - aborting")
                    return False
            
            # Move to piece height with slower feedrate for precision
            self.logger.info(f"Moving down to king height {king_settings['height']}")
            if not self.move_nozzle_smooth([('Z', king_settings['height'] - self.SAFE_HEIGHT)], 
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to king height")
                return False
                
            # Grip the king
            self.logger.info(f"Gripping king with pulse width {king_settings['grip_pw']}")
            if not self.set_gripper_position(king_settings['grip_pw'], slow_mode=True):
                self.logger.error("Failed to grip king")
                # Emergency move back up to safe height
                self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - king_settings['height'])])
                return False
                
            # Extra delay to ensure grip is secure
            time.sleep(0.05)  # Increased delay for reliability
            
            # Verify gripper is actually gripping
            if not self.verify_gripper_state("closed"):
                self.logger.warning("Gripper not in closed state after gripping - forcing state update")
                self.gripper_state = "closed"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - king_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after pickup")
                # Try emergency move to safe height
                if not self.move_to_z_height(self.SAFE_HEIGHT):
                    self.logger.error("Emergency height recovery failed - trying to release king anyway")
                    self.emergency_release()
                    return False
                
            # Move to king target square
            self.logger.info(f"Moving to king target square {king_target} (display: {display_king_target})")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position before target move")
                self.emergency_release()
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', king_target_pos['x'] - current_pos['X']), 
                ('Y', king_target_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to king target position")
                self.emergency_release()
                return False
            
            # Move down to piece placement height
            self.logger.info("Moving down to place king")
            if not self.move_nozzle_smooth([('Z', king_settings['height'] - self.SAFE_HEIGHT)],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to king placement height")
                # Try to get to a reasonable height
                self.move_to_z_height(king_target_pos['z'] + 20)  # Slightly above target
                self.emergency_release()
                return False
            
            # Release the king using slow mode for controlled release
            self.logger.info("Releasing king")
            if not self.open_gripper(slow_mode=True):
                self.logger.error("Failed to release king normally - attempting emergency release")
                self.emergency_release()
                # Continue with move regardless
            
            # Extra delay to ensure complete release
            time.sleep(0.05)  # Increased delay for reliability
            
            # Verify gripper is actually open
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open after release - forcing state update")
                self.gripper_state = "open"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height after placing king")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - king_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after release")
                # Try emergency move to safe height
                self.move_to_z_height(self.SAFE_HEIGHT)
                # Continue with move regardless
            
            ########################## ROOK MOVEMENT ##########################
            self.logger.info("===== ROOK MOVEMENT PHASE =====")
                
            # Get rook piece information
            rook_piece = self.chess_game.get_piece_on_square(rook_source)
            if not rook_piece:
                self.logger.error(f"No rook found on {rook_source} (display: {display_rook_source})")
                return False
                
            # Get rook settings
            rook_settings = self.piece_mapper.get_piece_settings(rook_piece.type.value)
            if not rook_settings:
                self.logger.error("No calibration settings found for the rook!")
                return False
                
            # Verify gripper is still open before moving rook
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open before rook pickup - forcing open")
                if not self.open_gripper(force=True):
                    self.logger.error("Failed to ensure gripper is open before rook pickup - aborting")
                    return False
            
            # Move to rook source square
            self.logger.info(f"Moving to rook source square {rook_source} (display: {display_rook_source})")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position")
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', rook_source_pos['x'] - current_pos['X']), 
                ('Y', rook_source_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to rook source position")
                return False
            
            # Verify gripper is still open before descending
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open before pickup - forcing open")
                if not self.open_gripper(force=True):
                    self.logger.error("Failed to ensure gripper is open before pickup - aborting")
                    return False
            
            # Move to piece height with slower feedrate for precision
            self.logger.info(f"Moving down to rook height {rook_settings['height']}")
            if not self.move_nozzle_smooth([('Z', rook_settings['height'] - self.SAFE_HEIGHT)], 
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to rook height")
                return False
                
            # Grip the rook
            self.logger.info(f"Gripping rook with pulse width {rook_settings['grip_pw']}")
            if not self.set_gripper_position(rook_settings['grip_pw'], slow_mode=True):
                self.logger.error("Failed to grip rook")
                # Emergency move back up to safe height
                self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - rook_settings['height'])])
                return False
                
            # Extra delay to ensure grip is secure
            time.sleep(0.05)  # Increased delay for reliability
            
            # Verify gripper is actually gripping
            if not self.verify_gripper_state("closed"):
                self.logger.warning("Gripper not in closed state after gripping - forcing state update")
                self.gripper_state = "closed"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - rook_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after pickup")
                # Try emergency move to safe height
                if not self.move_to_z_height(self.SAFE_HEIGHT):
                    self.logger.error("Emergency height recovery failed - trying to release rook anyway")
                    self.emergency_release()
                    return False
                
            # Move to rook target square
            self.logger.info(f"Moving to rook target square {rook_target} (display: {display_rook_target})")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position before target move")
                self.emergency_release()
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', rook_target_pos['x'] - current_pos['X']), 
                ('Y', rook_target_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to rook target position")
                self.emergency_release()
                return False
            
            # Move down to piece placement height
            self.logger.info("Moving down to place rook")
            if not self.move_nozzle_smooth([('Z', rook_settings['height'] - self.SAFE_HEIGHT)],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to rook placement height")
                # Try to get to a reasonable height
                self.move_to_z_height(rook_target_pos['z'] + 20)  # Slightly above target
                self.emergency_release()
                return False
            
            # Release the rook using slow mode for controlled release
            self.logger.info("Releasing rook")
            if not self.open_gripper(slow_mode=True):
                self.logger.error("Failed to release rook normally - attempting emergency release")
                self.emergency_release()
                # Continue with move regardless
            
            # Extra delay to ensure complete release
            time.sleep(0.05)  # Increased delay for reliability
            
            # Verify gripper is actually open
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open after release - forcing state update")
                self.gripper_state = "open"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height after placing rook")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - rook_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after release")
                # Try emergency move to safe height
                self.move_to_z_height(self.SAFE_HEIGHT)
                # Continue with move regardless
            
            # Update game state
            self.chess_game.board.push(move_obj)
            self.current_turn = self.chess_game.get_turn()
            self.show_status()
            
            # Log information about the castling move with user-friendly square names
            self.logger.info(f"===== CASTLING COMPLETED SUCCESSFULLY =====")
            self.logger.info(f"King moved: {display_king_source} → {display_king_target}")
            self.logger.info(f"Rook moved: {display_rook_source} → {display_rook_target}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing castling move {notation}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Emergency release in case we're holding a piece
            try:
                self.emergency_release()
            except Exception as release_err:
                self.logger.error(f"Error during emergency release: {str(release_err)}")
            
            # Try to get back to safe height
            try:
                self.move_to_z_height(self.SAFE_HEIGHT)
            except:
                pass
                
            return False
           
    def _execute_direct_move(self, source_square: str, target_square: str) -> bool:
        """
        Execute a move by directly specifying source and target squares.
        Updated to match the play_move approach.
        """
        try:
            # Get the piece details
            piece = self.chess_game.get_piece_on_square(source_square)
            if not piece:
                self.logger.error(f"No piece found on square {source_square}")
                return False
                
            piece_settings = self.piece_mapper.get_piece_settings(piece.type.value)
            if not piece_settings:
                self.logger.error(f"No settings found for {piece.type.value}")
                return False
            
            # CRITICAL SAFETY CHECK: Ensure gripper is open before starting any move
            if not self.verify_gripper_state("open"):
                self.logger.error("Cannot guarantee gripper is open after verification - retrying once")
                # One more try with forced open
                success = self.open_gripper(force=True)
                if not success:
                    self.logger.error("Failed to ensure gripper is open even after retry - aborting move")
                    return False
            
            # Check if this is a capture
            is_capture = self.chess_game.is_capture(target_square)
            if is_capture:
                captured_piece = self.chess_game.get_piece_on_square(target_square)
                if captured_piece:
                    # Determine proper storage box based on piece color and board orientation
                    if self.chess_mapper.flipped:
                        # When board is flipped (playing as black), reverse the box choices
                        storage_box = 'box_1' if captured_piece.color == Color.WHITE else 'box_2'
                    else:
                        # Standard orientation (playing as white) 
                        storage_box = 'box_2' if captured_piece.color == Color.WHITE else 'box_1'
                    self.logger.info(f"Capturing {captured_piece.type.value} on {target_square} to {storage_box}")
                    capture_success = self._capture_piece(target_square, storage_box)
                    if not capture_success:
                        self.logger.error("Failed to capture piece - aborting move")
                        return False
                    
                    # Make sure gripper is open again after capture
                    if not self.verify_gripper_state("open"):
                        self.logger.warning("Gripper not open after capture - forcing open")
                        if not self.open_gripper(force=True):
                            self.logger.error("Failed to open gripper after capture - aborting move")
                            return False
            
            # Get source square coordinates
            source_pos = self.get_transformed_position(source_square)
            if not source_pos:
                self.logger.error(f"No position mapping for square {source_square}")
                return False
            
            # Move to safe height first
            self.logger.info(f"Moving to safe height {self.SAFE_HEIGHT}")
            if not self.move_to_z_height(self.SAFE_HEIGHT):
                self.logger.error("Failed to move to safe height")
                return False
            
            # Move to source square
            self.logger.info(f"Moving to source square {source_square}")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position")
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', source_pos['x'] - current_pos['X']), 
                ('Y', source_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to source square position")
                return False
            
            # Verify gripper is still open before descending
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open before pickup - forcing open")
                if not self.open_gripper(force=True):
                    self.logger.error("Failed to ensure gripper is open before pickup - aborting")
                    return False
            
            # Move to piece height with slower feedrate for precision
            self.logger.info(f"Moving down to piece height {piece_settings['height']}")
            if not self.move_nozzle_smooth([('Z', piece_settings['height'] - self.SAFE_HEIGHT)], 
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to piece height")
                return False
                
            # Grip the piece
            self.logger.info(f"Gripping piece with pulse width {piece_settings['grip_pw']}")
            if not self.set_gripper_position(piece_settings['grip_pw']):
                self.logger.error("Failed to grip piece")
                # Emergency move back up to safe height
                self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])])
                return False
                
            # Extra delay to ensure grip is secure
            time.sleep(0.02)
            
            # Verify gripper is actually gripping
            if not self.verify_gripper_state("closed"):
                self.logger.warning("Gripper not in closed state after gripping - forcing state update")
                self.gripper_state = "closed"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after pickup")
                # Try emergency move to safe height
                if not self.move_to_z_height(self.SAFE_HEIGHT):
                    self.logger.error("Emergency height recovery failed - trying to release piece anyway")
                    self.emergency_release()
                    return False
                
            # Get target square coordinates
            target_pos = self.get_transformed_position(target_square)
            if not target_pos:
                self.logger.error(f"No position mapping for target square {target_square}")
                # Release piece since we can't place it properly
                self.emergency_release()
                return False
                
            # Move to target square
            self.logger.info(f"Moving to target square {target_square}")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position before target move")
                self.emergency_release()
                return False
                
            # Move X and Y simultaneously
            if not self.move_nozzle_smooth([
                ('X', target_pos['x'] - current_pos['X']), 
                ('Y', target_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error("Failed to move to target square position")
                self.emergency_release()
                return False
            
            # Move down to piece placement height
            self.logger.info("Moving down to place piece")
            if not self.move_nozzle_smooth([('Z', piece_settings['height'] - self.SAFE_HEIGHT)],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to piece placement height")
                # Try to get to a reasonable height
                self.move_to_z_height(target_pos['z'] + 20)  # Slightly above target
                self.emergency_release()
                return False
            
            # Release the piece using slow mode for controlled release
            self.logger.info("Releasing piece")
            if not self.open_gripper(slow_mode=True):
                self.logger.error("Failed to release piece normally - attempting emergency release")
                self.emergency_release()
                # Continue with move regardless
            
            # Extra delay to ensure complete release
            time.sleep(0.02)
            
            # Verify gripper is actually open
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open after release - forcing state update")
                self.gripper_state = "open"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height after placing piece")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after release")
                # Try emergency move to safe height
                self.move_to_z_height(self.SAFE_HEIGHT)
                # Continue with move regardless
            
            # Update game state
            try:
                move = chess.Move.from_uci(f"{source_square}{target_square}")
                self.chess_game.board.push(move)
                self.current_turn = self.chess_game.get_turn()
                self.show_status()
                
                self.logger.info(f"Successfully executed direct move: {source_square} to {target_square}")
                return True
            except ValueError as e:
                self.logger.error(f"Error updating game state: {str(e)}")
                # Even if we can't update the state, we moved the piece physically
                self.logger.info(f"Physical move completed: {source_square} to {target_square}")
                return True
        
        except Exception as e:
            self.logger.error(f"Error executing direct move {source_square} to {target_square}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Emergency release in case we're holding a piece
            try:
                self.emergency_release()
            except Exception as release_err:
                self.logger.error(f"Error during emergency release: {str(release_err)}")
            
            # Try to get back to safe height
            try:
                self.move_to_z_height(self.SAFE_HEIGHT)
            except:
                pass
                
            return False

    def _capture_piece(self, square: str, storage_location: str) -> bool:
        """Enhanced method to capture and store a chess piece with better error handling."""
        try:
            # Get the piece details
            piece = self.chess_game.get_piece_on_square(square)
            if not piece:
                self.logger.error(f"No piece found on square {square}")
                return False
                
            piece_settings = self.piece_mapper.get_piece_settings(piece.type.value)
            if not piece_settings:
                self.logger.error(f"No settings found for {piece.type.value}")
                return False
            
            # CRITICAL SAFETY CHECK: Ensure gripper is open before starting any move
            if not self.verify_gripper_state("open"):
                self.logger.error("Cannot guarantee gripper is open after verification - retrying once")
                # One more try with forced open
                success = self.open_gripper(force=True)
                if not success:
                    self.logger.error("Failed to ensure gripper is open even after retry - aborting capture")
                    return False
            
            # Move to safe height first
            self.logger.info(f"Moving to safe height {self.SAFE_HEIGHT}")
            if not self.move_to_z_height(self.SAFE_HEIGHT):
                self.logger.error("Failed to move to safe height")
                return False
            
            # Get square coordinates
            square_pos = self.get_transformed_position(square)
            if not square_pos:
                self.logger.error(f"No position mapping for square {square}")
                return False
                
            # Move to square (X and Y simultaneously)
            self.logger.info(f"Moving to source square {square}")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position")
                return False
                
            if not self.move_nozzle_smooth([
                ('X', square_pos['x'] - current_pos['X']), 
                ('Y', square_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error(f"Failed to move to square {square}")
                return False
            
            # Verify gripper is still open before descending
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open before pickup - forcing open")
                if not self.open_gripper(force=True):
                    self.logger.error("Failed to ensure gripper is open before pickup - aborting")
                    return False
                
            # Move directly to piece height from safe height with slower feedrate
            self.logger.info(f"Moving down to piece height {piece_settings['height']}")
            if not self.move_nozzle_smooth([('Z', piece_settings['height'] - self.SAFE_HEIGHT)], 
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to piece height")
                return False
            
            # Grip the piece
            self.logger.info(f"Gripping piece with pulse width {piece_settings['grip_pw']}")
            if not self.set_gripper_position(piece_settings['grip_pw'], slow_mode=True):
                self.logger.error("Failed to grip piece")
                # Emergency move back up to safe height
                self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])])
                return False
            
            # Extra delay to ensure grip is secure
            time.sleep(0.02)
            
            # Verify gripper is actually gripping
            if not self.verify_gripper_state("closed"):
                self.logger.warning("Gripper not in closed state after gripping - forcing state update")
                self.gripper_state = "closed"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - piece_settings['height'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after pickup")
                # Try emergency move to safe height
                if not self.move_to_z_height(self.SAFE_HEIGHT):
                    self.logger.error("Emergency height recovery failed - trying to release piece anyway")
                    self.emergency_release()
                    return False
            
            # Get storage location coordinates
            storage_pos = self.storage_mapper.get_position(storage_location)
            if not storage_pos:
                self.logger.error(f"No position mapping for storage {storage_location}")
                # Release piece since we can't place it properly
                self.emergency_release()
                return False
                
            # Move to storage (X and Y simultaneously)
            self.logger.info(f"Moving to storage location {storage_location}")
            current_pos = self.get_position()
            if not current_pos:
                self.logger.error("Failed to get current position before storage move")
                self.emergency_release()
                return False
                
            if not self.move_nozzle_smooth([
                ('X', storage_pos['x'] - current_pos['X']), 
                ('Y', storage_pos['y'] - current_pos['Y'])
            ]):
                self.logger.error(f"Failed to move to storage {storage_location}")
                self.emergency_release()
                return False
            
            # Move down to storage height
            self.logger.info(f"Moving down to storage height {storage_pos['z']}")
            if not self.move_nozzle_smooth([('Z', storage_pos['z'] - self.SAFE_HEIGHT)],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move to storage height")
                # Try to get to a reasonable height
                self.move_to_z_height(storage_pos['z'] + 20)  # Slightly above target
                self.emergency_release()
                return False
            
            # Release the piece using slow mode for controlled release
            self.logger.info("Releasing piece")
            if not self.open_gripper(slow_mode=True):
                self.logger.error("Failed to release piece normally - attempting emergency release")
                self.emergency_release()
                # Continue with move regardless
            
            # Extra delay to ensure complete release
            time.sleep(0.02)
            
            # Verify gripper is actually open
            if not self.verify_gripper_state("open"):
                self.logger.warning("Gripper not open after release - forcing state update")
                self.gripper_state = "open"
            
            # Move back up to safe height
            self.logger.info("Moving back to safe height after storing piece")
            if not self.move_nozzle_smooth([('Z', self.SAFE_HEIGHT - storage_pos['z'])],
                                          feedrate=self.config.z_feedrate // 2):
                self.logger.error("Failed to move back to safe height after release")
                # Try emergency move to safe height
                self.move_to_z_height(self.SAFE_HEIGHT)
                # Continue with move regardless
            
            # Success
            self.logger.info(f"Successfully captured piece from {square} to {storage_location}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during capture sequence: {str(e)}")
            self.logger.debug(traceback.format_exc())
            
            # Emergency release if an error occurs
            try:
                self.emergency_release()
            except Exception as release_err:
                self.logger.error(f"Error during emergency release: {str(release_err)}")
            
            # Try to get back to safe height
            try:
                self.move_to_z_height(self.SAFE_HEIGHT)
            except:
                pass
                
            return False
       
    def play_against_stockfish(self) -> None:
        """Play a game against Stockfish with enhanced gripper safety and fixed duplicate verifications."""
        try:
            if not self.chess_game.engine:
                self.logger.error("Stockfish engine not available!")
                return
                
            print("\nStockfish Game Setup")
            while True:
                color = input("Choose your color (white/black): ").strip().lower()
                if color in ['white', 'black']:
                    break
                print("Please enter 'white' or 'black'")
                
            difficulty = self.select_difficulty()
                
            while True:
                movement = input("Do you want to move your pieces manually? (yes/no): ").strip().lower()
                if movement in ['yes', 'no']:
                    manual_movement = movement == 'yes'
                    break
                print("Please enter 'yes' or 'no'")
                
            verify_option = input("Enable board state verification after complex moves? (yes/no): ").strip().lower()
            verify_enabled = verify_option == 'yes'

            gripper_check = input("Enable periodic gripper state checks? (yes/no): ").strip().lower()
            gripper_check_enabled = gripper_check == 'yes'
            
            self.chess_game = ChessGame()
            self.current_turn = Color.WHITE
            player_is_white = color == 'white'
            
            # Set board orientation based on player color
            self.set_board_orientation(player_is_white)
            
            print("\nStockfish Game Mode")
            print(f"You play as {color.capitalize()}")
            print(f"Board orientation: {'Standard (White)' if player_is_white else 'Flipped (Black)'}")
            print("Board verification:", "Enabled" if verify_enabled else "Disabled")
            print("Commands:")
            print("  'show'   - Display current position")
            print("  'help'   - Display move format help")
            print("  'legal'  - Show all legal moves")
            print("  'verify' - Manually verify board state")
            print("  'gripper'- Check and reset gripper state")
            print("  'uci'    - Show the move you want to make in UCI format")
            print("  'exit'   - End game and return to main menu")
            
            if manual_movement:
                print("\nIn manual mode:")
                print("1. Board will move forward for access")
                print("2. Make your move on the board")
                print("3. Enter the move you made in algebraic notation (e.g., e4, Nf3)")
                print("4. Board will return and Stockfish will move")
                    
            self.chess_game.display_state(self.chess_mapper.flipped)
            
            # Add safety check at start of game - ensure gripper is open
            if gripper_check_enabled:
                self.logger.info("Initial gripper safety check")
                self.verify_gripper_state("open")
                
            # Initialize last verification time to avoid redundant checks
            self._last_verification_time = 0
            
            # Track complex moves for verification and move counter for gripper checks
            last_move_was_complex = False
            last_verified_move_count = 0  # Track the last move where we performed verification
            move_counter = 0
                
            while True:
                # Verify board state after complex moves if enabled, but only if we haven't verified recently
                if verify_enabled and last_move_was_complex and move_counter > last_verified_move_count:
                    current_time = time.time()
                    # Only verify if it's been more than 5 seconds since last verification
                    if not hasattr(self, '_last_verification_time') or current_time - self._last_verification_time > 5:
                        if not self.verify_board_state():
                            print("Board state verification failed. Please use 'verify' command to fix discrepancies.")
                        last_verified_move_count = move_counter  # Update the last verified move count
                    last_move_was_complex = False

                # Periodic gripper check every 5 moves if enabled
                if gripper_check_enabled and move_counter % 5 == 0 and move_counter > 0:
                    self.logger.info(f"Periodic gripper check (move {move_counter})")
                    self.verify_gripper_state("open")
                        
                is_player_turn = (player_is_white and self.chess_game.board.turn) or \
                                 (not player_is_white and not self.chess_game.board.turn)
                                     
                if is_player_turn:
                    while True:
                        if manual_movement:
                            print("\nMoving board forward for access...")
                            self.move_to_board_access_position()
                            print("\nMake your move on the board, then enter the move in algebraic notation")
                            move = input("Enter move or command: ").strip()
                        else:
                            move = input("\nEnter your move: ").strip()
                                
                        # Command handling
                        if move.lower() == 'exit':
                            return
                        elif move.lower() == 'show':
                            self.chess_game.display_state(self.chess_mapper.flipped)
                            continue
                        elif move.lower() == 'help':
                            self.print_move_help()
                            continue
                        elif move.lower() == 'legal':
                            self.validate_legal_moves()
                            continue
                        elif move.lower() == 'verify':
                            self.verify_board_state()
                            # Update verification timestamp and counter
                            self._last_verification_time = time.time()
                            last_verified_move_count = move_counter
                            continue
                        elif move.lower() == 'gripper':
                            print("Checking gripper state...")
                            self.verify_gripper_state("open")
                            print("Gripper check complete")
                            continue
                        elif move.lower() == 'uci':
                            piece_from = input("Enter the starting square (e.g., e2): ").strip().lower()
                            piece_to = input("Enter the destination square (e.g., e4): ").strip().lower()
                                
                            if len(piece_from) != 2 or len(piece_to) != 2:
                                print("Invalid square format. Use format like 'e2' or 'g8'.")
                                continue
                                    
                            move = piece_from + piece_to
                        elif not move:
                            continue
                            
                        # Check if this is a complex move (castling, capture, etc.)
                        is_complex_move = 'O-O' in move or 'x' in move
                            
                        # Try multiple parsing approaches
                        squares = self.chess_game.parse_move(move)
                        if squares:
                            source_square, target_square = squares
                                
                            if manual_movement:
                                # Just update internal state for manual movement
                                self.chess_game.update_position(source_square, target_square)
                                # Set flag for complex moves
                                last_move_was_complex = is_complex_move
                                # Increment move counter
                                move_counter += 1
                                break
                            else:
                                # Check gripper state before moving
                                if gripper_check_enabled:
                                    self.verify_gripper_state("open")
                                    
                                # Move the physical pieces
                                if self.play_move(move):
                                    # Set flag for complex moves
                                    last_move_was_complex = is_complex_move
                                    # Increment move counter
                                    move_counter += 1
                                    break
                                else:
                                    # If standard notation fails, try with UCI format
                                    print("Failed to execute move! Trying alternative formats...")
                                    try:
                                        # Try direct UCI format (source to target)
                                        if self.play_move(f"{source_square}{target_square}"):
                                            print(f"Successfully executed move using UCI format: {source_square}{target_square}")
                                            last_move_was_complex = is_complex_move
                                            # Increment move counter
                                            move_counter += 1
                                            break
                                    except Exception as move_error:
                                        self.logger.error(f"Error with alternative move format: {str(move_error)}")
                                        
                                    print("All move formats failed. Please try again or type 'help' for guidance.")
                        else:
                            print("Invalid move! Try again or type 'legal' to see valid moves.")
                else:
                    print("\nStockfish is thinking...")
                    
                    # Get engine configuration
                    config = getattr(self.chess_game, 'engine_config', {
                        "time_limit": 0.1,
                        "depth_limit": None
                    })
                    
                    # Get evaluation with configured limits
                    result = self.chess_game.engine.play(
                        self.chess_game.board, 
                        chess.engine.Limit(
                            time=config.get("time_limit", 0.1),
                            depth=config.get("depth_limit")
                        )
                    )

                    engine_move = self.chess_game.board.san(result.move)
                    print(f"\nStockfish plays: {engine_move}")

                    # Check if the move results in checkmate
                    is_checkmate = False
                    if "#" in engine_move:
                        is_checkmate = True
                        print("Checkmate!")
                    elif "+" in engine_move:
                        print("Check!")
                        
                    # Check if this is a complex move
                    is_complex_move = 'O-O' in engine_move or 'x' in engine_move or engine_move[0].lower() in 'abcdefgh'
                        
                    # Extract source and target squares for the move
                    source = chess.square_name(result.move.from_square)
                    target = chess.square_name(result.move.to_square)

                    # Check gripper state before Stockfish moves
                    if gripper_check_enabled:
                        self.verify_gripper_state("open")
                        
                    # Try multiple approaches to execute the move
                    success = False
                        
                    # Try SAN notation first
                    if self.play_move(engine_move):
                        success = True
                        last_move_was_complex = is_complex_move
                        # Increment move counter
                        move_counter += 1
                    # Try UCI format as fallback
                    elif self.play_move(f"{source}{target}"):
                        success = True
                        last_move_was_complex = is_complex_move
                        # Increment move counter
                        move_counter += 1
                    # Try decomposed move as last resort
                    elif not success:
                        try:
                            print(f"Trying direct execution via update_position: {source} to {target}")
                            # Update internal state
                            self.chess_game.update_position(source, target)
                            # Try our own implementation
                            self._execute_direct_move(source, target)
                            success = True
                            last_move_was_complex = is_complex_move
                            # Increment move counter
                            move_counter += 1
                        except Exception as e:
                            self.logger.error(f"Final fallback execution also failed: {str(e)}")
                        
                    if not success:
                        self.logger.error(f"Failed to execute Stockfish's move: {engine_move}")
                        print("Error executing Stockfish's move. Please make the following move manually:")
                        print(f"Move piece from {source} to {target}")
                        input("Press Enter after making the move...")
                        # Update internal state
                        self.chess_game.update_position(source, target)
                        # Always verify after manual intervention
                        if verify_enabled:
                            self.verify_board_state()
                            # Update verification timestamp and counter
                            self._last_verification_time = time.time()
                            last_verified_move_count = move_counter
                            
                        # Reset complex move flag
                        last_move_was_complex = False
                        # Increment move counter
                        move_counter += 1

                    # After any move (successful or not), ensure gripper is open
                    if gripper_check_enabled:
                        self.verify_gripper_state("open")
                            
                    # Special additional check for pawn moves
                    if success and engine_move[0].lower() in 'abcdefgh':
                        print("Stockfish made a pawn move. Verifying source square is now empty...")
                        empty_verification = input(f"Is square {source} now empty? (yes/no): ").strip().lower()
                        if empty_verification != 'yes':
                            print("Source square should be empty after a pawn move. Please check the board state.")
                            if verify_enabled:
                                self.verify_board_state()
                                # Update verification timestamp and counter
                                self._last_verification_time = time.time()
                                last_verified_move_count = move_counter
                            
                    # Additional verification for complex moves
                    if success and is_complex_move and verify_enabled:
                        # Enough time since last verification?
                        current_time = time.time()
                        if not hasattr(self, '_last_verification_time') or current_time - self._last_verification_time > 5:
                            print(f"Complex move detected: {engine_move}")
                            print("Performing additional verification...")
                            if not self.verify_board_state():
                                print("Board state verification failed after complex move.")
                            # Update verification timestamp and counter
                            self._last_verification_time = time.time()
                            last_verified_move_count = move_counter
                                
                if self.chess_game.board.is_game_over():
                    self._show_game_result()
                    break
                        
        except KeyboardInterrupt:
            print("\nGame interrupted!")
        except Exception as e:
            self.logger.error(f"Error in Stockfish game mode: {str(e)}")
            traceback.print_exc()
        finally:
            # Final safety check - make sure gripper is open
            try:
                if gripper_check_enabled:
                    self.logger.info("Final gripper safety check")
                    self.verify_gripper_state("open")
            except:
                pass
                    
            if manual_movement:
                self.move_to_board_access_position()
            print("\nReturning to main menu...")
    
    def select_difficulty(self):
        """
        Interactive menu to select difficulty level and configure engine strength.
        
        Returns:
            str: Selected difficulty level name
        """
        print("\nSelect Difficulty Level:")
        print("┌──────────────────────────────────────────────────┐")
        print("│ 1. Absolute Beginner (Elo ~600)                  │")
        print("│    Making basic errors, missing simple captures  │")
        print("│                                                  │")
        print("│ 2. Beginner (Elo ~900)                           │")
        print("│    Plays like a novice who knows the rules       │")
        print("│                                                  │")
        print("│ 3. Casual (Elo ~1300)                            │")
        print("│    Occasional tactical errors                    │")
        print("│                                                  │")
        print("│ 4. Intermediate (Elo ~1600)                      │")
        print("│    Solid play with some positional understanding │")
        print("│                                                  │")
        print("│ 5. Club Player (Elo ~1900)                       │")
        print("│    Strong tactical play                          │")
        print("│                                                  │")
        print("│ 6. Advanced (Elo ~2100)                          │")
        print("│    Tournament-level strength                     │")
        print("│                                                  │")
        print("│ 7. Expert (Elo 2400+)                            │")
        print("│    Near-master level play                        │")
        print("└──────────────────────────────────────────────────┘")
        
        # Difficulty presets mapping to Stockfish parameters
        strength_presets = {
            '1': {"name": "Absolute Beginner", "skill": 0, "depth": 1, "time": 0.05, "elo": "~600"},
            '2': {"name": "Beginner", "skill": 3, "depth": 2, "time": 0.1, "elo": "~900"},
            '3': {"name": "Casual", "skill": 6, "depth": 3, "time": 0.3, "elo": "~1300"},
            '4': {"name": "Intermediate", "skill": 10, "depth": 5, "time": 0.5, "elo": "~1600"},
            '5': {"name": "Club Player", "skill": 13, "depth": 8, "time": 0.8, "elo": "~1900"},
            '6': {"name": "Advanced", "skill": 16, "depth": 12, "time": 1.0, "elo": "~2100"},
            '7': {"name": "Expert", "skill": 20, "depth": None, "time": 2.0, "elo": "2400+"}
        }
        
        while True:
            choice = input("Choose difficulty (1-7): ").strip()
            if choice in strength_presets:
                selected = strength_presets[choice]
                
                # Configure the chess engine
                self.chess_game.configure_strength(
                    skill_level=selected["skill"],
                    depth_limit=selected["depth"],
                    time_limit=selected["time"]
                )
                
                print(f"\nSelected: {selected['name']} (Approx. Elo {selected['elo']})")
                return selected["name"]
            else:
                print("Please enter a number from 1 to 7")

    def move_to_board_access_position(self) -> bool:
        """
        Move printer to a position where the board is accessible for manual moves.
        Uses consistent class constants for positioning.
        """
        try:
            # First move Z to safe height
            if not self.move_to_z_height(self.SAFE_HEIGHT):
                return False
                
            # Then move Y to access position
            current_pos = self.get_position()
            if current_pos:
                if not self.move_nozzle_smooth([('Y', self.ACCESS_Y - current_pos['Y'])]):
                    return False
                    
            self.logger.info(f"Moved to board access position (Z={self.SAFE_HEIGHT}, Y={self.ACCESS_Y})")
            return True
        except Exception as e:
            self.logger.error(f"Error moving to board access position: {str(e)}")
            return False
        
    def validate_legal_moves(self) -> None:
        """Display all legal moves in the current position with explanations."""
        try:
            # Get all legal moves in SAN format
            legal_moves = [self.chess_game.board.san(move) for move in self.chess_game.board.legal_moves]
            
            if legal_moves:
                print("\nLegal moves in current position:")
                
                # Organize moves by type for better readability
                pawn_moves = []
                piece_moves = []
                captures = []
                special = []
                
                for move in legal_moves:
                    if 'x' in move:
                        captures.append(move)
                    elif move in ['O-O', 'O-O-O']:
                        special.append(move)
                    elif move[0].isupper():
                        piece_moves.append(move)
                    else:
                        pawn_moves.append(move)
                
                if pawn_moves:
                    print("\n  Pawn moves:", ", ".join(pawn_moves))
                if piece_moves:
                    print("  Piece moves:", ", ".join(piece_moves))
                if captures:
                    print("  Captures:", ", ".join(captures))
                if special:
                    print("  Special moves:", ", ".join(special))
                    
                # Also show UCI format for a few examples
                print("\nSome examples in UCI format (source square to target square):")
                uci_examples = [move.uci() for move in list(self.chess_game.board.legal_moves)[:5]]
                print("  " + ", ".join(uci_examples))
                
            else:
                print("\nNo legal moves available - game is over")
            
        except Exception as e:
            self.logger.error(f"Error displaying legal moves: {str(e)}")
            traceback.print_exc()

    def validate_game_state(self):
        try:
            fen = self.chess_game.board.fen()
            expected_color = Color.WHITE if self.chess_game.board.turn else Color.BLACK
            if self.current_turn != expected_color:
                self.logger.warning(f"Turn mismatch detected. Fixing from {self.current_turn} to {expected_color}")
                self.current_turn = expected_color
            castling_rights = self.chess_game.board.castling_rights
            if not castling_rights:
                self.logger.info("No remaining castling rights")
            legal_moves = [self.chess_game.board.san(move) for move in self.chess_game.board.legal_moves]
            self.logger.info(f"Valid moves in current position: {', '.join(legal_moves)}")
            return True
        except Exception as e:
            self.logger.error(f"Error validating game state: {str(e)}")
            return False

    def verify_board_state(self) -> bool:
        """
        Verify that the internal board state matches the physical board.
        Returns True if verification is successful, False otherwise.
        """
        try:
            # Update the verification timestamp
            self._last_verification_time = time.time()
            
            # Ask user to confirm board state after complex moves
            print("\nBoard State Verification:")
            if self.chess_mapper.flipped:
                print("NOTE: Board is in BLACK perspective (flipped coordinates)")
            print("Please confirm the physical board matches the displayed position.")
        
            # Display the current board state
            self.chess_game.display_state(self.chess_mapper.flipped)
            
            # Ask for confirmation
            while True:
                confirmation = input("Does the physical board match this state? (yes/no/fix): ").strip().lower()
                
                if confirmation == "yes":
                    self.logger.info("Board state verification confirmed by user")
                    return True
                elif confirmation == "no":
                    print("Board state mismatch detected.")
                    return False
                elif confirmation == "fix":
                    self._fix_board_state()
                    return self.verify_board_state()  # Recursively verify after fixing
                else:
                    print("Please enter 'yes', 'no', or 'fix'")
        
        except Exception as e:
            self.logger.error(f"Error during board state verification: {str(e)}")
            traceback.print_exc()
            return False

    def _fix_board_state(self) -> None:
        """
        Interactive utility to fix discrepancies between internal and physical board state.
        """
        try:
            print("\nBoard State Correction Mode")
            print("Options:")
            print("  1. Manual correction (physically move pieces)")
            print("  2. Adjust internal state")
            print("  3. Return without making changes")
            
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == "1":
                print("\nManual Correction:")
                print("1. The board will move to access position")
                print("2. Adjust the physical pieces to match the displayed state")
                print("3. Confirm when done")
                
                # Move board to accessible position
                self.move_to_board_access_position()
                
                input("Press Enter when you've finished adjusting the physical pieces...")
                return
                
            elif choice == "2":
                print("\nInternal State Adjustment:")
                print("You can manually update the internal board state to match physical pieces.")
                
                while True:
                    print("\nOptions:")
                    print("  'set square piece' - Set a piece on a square (e.g., 'set e4 wp' for white pawn)")
                    print("  'clear square'     - Remove a piece from a square (e.g., 'clear e4')")
                    print("  'fen'              - Set the complete board using FEN notation")
                    print("  'done'             - Finish adjustment")
                    
                    cmd = input("Enter command: ").strip().lower()
                    
                    if cmd == "done":
                        break
                        
                    elif cmd.startswith("set "):
                        try:
                            _, square, piece_code = cmd.split()
                            if not re.match(r'^[a-h][1-8]$', square):
                                print("Invalid square notation. Use a1-h8.")
                                continue
                                
                            color = chess.WHITE if piece_code[0] == 'w' else chess.BLACK
                            piece_type_map = {'p': chess.PAWN, 'n': chess.KNIGHT, 'b': chess.BISHOP, 
                                             'r': chess.ROOK, 'q': chess.QUEEN, 'k': chess.KING}
                            piece_type = piece_type_map.get(piece_code[1].lower())
                            
                            if piece_type is None:
                                print("Invalid piece code. Use 'wp' for white pawn, 'bk' for black king, etc.")
                                continue
                                
                            # Create a new board with the current state
                            new_board = chess.Board(self.chess_game.board.fen())
                            
                            # Set the piece on the square
                            new_board.set_piece_at(chess.parse_square(square), chess.Piece(piece_type, color))
                            
                            # Replace the current board
                            self.chess_game.board = new_board
                            self.chess_game.display_state(self.chess_mapper.flipped)
                            
                        except ValueError:
                            print("Invalid format. Use 'set square piece' (e.g., 'set e4 wp')")
                            
                    elif cmd.startswith("clear "):
                        try:
                            _, square = cmd.split()
                            if not re.match(r'^[a-h][1-8]$', square):
                                print("Invalid square notation. Use a1-h8.")
                                continue
                                
                            # Create a new board with the current state
                            new_board = chess.Board(self.chess_game.board.fen())
                            
                            # Remove the piece from the square
                            new_board.remove_piece_at(chess.parse_square(square))
                            
                            # Replace the current board
                            self.chess_game.board = new_board
                            self.chess_game.display_state(self.chess_mapper.flipped)
                            
                        except ValueError:
                            print("Invalid format. Use 'clear square' (e.g., 'clear e4')")
                            
                    elif cmd.startswith("fen "):
                        try:
                            fen = cmd[4:]  # Remove "fen " prefix
                            new_board = chess.Board(fen)
                            self.chess_game.board = new_board
                            self.chess_game.display_state(self.chess_mapper.flipped)
                        except ValueError as e:
                            print(f"Invalid FEN notation: {e}")
                    
                    else:
                        print("Unknown command. Please try again.")
                
            elif choice == "3":
                print("Returning without changes.")
                return
                
            else:
                print("Invalid choice. Returning without changes.")
                return
                
        except Exception as e:
            self.logger.error(f"Error during board state correction: {str(e)}")
            traceback.print_exc()

    def _show_game_result(self) -> None:
        print("\nGame Over!")
        if self.chess_game.board.is_checkmate():
            winner = "Black" if self.chess_game.board.turn == chess.WHITE else "White"
            print(f"Checkmate! {winner} wins!")
        elif self.chess_game.board.is_stalemate():
            print("Stalemate! Game is drawn.")
        elif self.chess_game.board.is_insufficient_material():
            print("Draw by insufficient material!")
        elif self.chess_game.board.is_fifty_moves():
            print("Draw by fifty-move rule!")
        elif self.chess_game.board.is_repetition():
            print("Draw by repetition!")
        else:
            print("Game ended in a draw!")

    def print_move_help(self):
        """Display comprehensive help information for chess move notation."""
        print("\nChess Notation Help:")
        print("\nPiece Moves:")
        print("  - King: K (e.g., Ke2)")
        print("  - Queen: Q (e.g., Qd4)")
        print("  - Rook: R (e.g., Rh1)")
        print("  - Bishop: B (e.g., Bf4)")
        print("  - Knight: N (e.g., Nf3)")
        print("  - Pawns: just the square (e.g., e4)")
        
        print("\nCaptures:")
        print("  - Piece captures: Include 'x' (e.g., Bxe5, Nxd4)")
        print("  - Pawn captures: File + x + square (e.g., exd5, bxc3)")
        
        print("\nSpecial Moves:")
        print("  - Castling kingside: O-O or 0-0")
        print("  - Castling queenside: O-O-O or 0-0-0")
        print("  - Promotion: e8Q (promote to queen)")
        
        print("\nAlternative Formats:")
        print("  - UCI format: Source + target (e.g., e2e4, b5c6)")
        print("  - When two pieces can move to same square:")
        print("    * Add file for disambiguation (e.g., Nbd2)")
        print("    * Or add rank (e.g., R1a3)")
        
        print("\nTips:")
        print("  - Enter 'show' to see the current board position")
        print("  - Enter 'legal' to see all legal moves")
        print("  - Enter 'exit' to end the game")

    
    # Calibration methods
    def recalibrate_z_offset(self) -> None:
        try:
            self.logger.info("Starting Z offset calibration...")
            self.is_calibrating = True 
            print("\nZ Offset Calibration:")
            print("1. Move the gripper down until it just touches the bed")
            print("2. This position will be set as Z=0")
            print("Use Z commands to adjust height (e.g., 'Z 0.1' or 'Z -0.1')")
            print("Enter 'done' when satisfied with position")
            print("Enter 'cancel' to abort calibration")
            while True:
                cmd = input("\nEnter Z adjustment or command: ").strip().lower()
                if cmd == 'done':
                    pos = self.get_position()
                    if pos:
                        confirm = input("\nSet current position as Z=0? (yes/no): ").strip().lower()
                        if confirm == 'yes':
                            self.send_gcode("G92 Z0")
                            self.z_offset = 0
                            self._save_z_offset()
                            self.logger.info("Z reference point established")
                            self.show_status()
                        else:
                            self.logger.info("Calibration cancelled")
                    break
                elif cmd == 'cancel':
                    self.logger.info("Calibration cancelled")
                    break
                elif cmd.startswith('z '):
                    try:
                        _, value = cmd.split()
                        value = float(value)
                        self.send_batched_commands([
                            "G91",
                            f"G1 Z{value} F{self.config.z_feedrate}",
                            "G90"
                        ])
                        self.show_status()
                    except ValueError:
                        self.logger.error("Invalid Z value")
                else:
                    self.logger.error("Invalid command. Use 'Z value', 'done', or 'cancel'")
        except Exception as e:
            self.logger.error(f"Z offset calibration failed: {str(e)}")
        finally:
            self.is_calibrating = False

    def _load_z_offset(self) -> None:
        try:
            if Path(self.config.z_offset_file).exists():
                with open(self.config.z_offset_file, 'r') as f:
                    data = json.load(f)
                    self.z_offset = data.get('z_offset')
                    if self.z_offset is not None:
                        self.logger.info(f"Loaded Z offset: {self.z_offset}mm")
                    else:
                        self.logger.warning("No Z offset found in file")
                        self.z_offset = None
            else:
                self.logger.info("No Z offset file found")
                self.z_offset = None
        except Exception as e:
            self.logger.error(f"Failed to load Z offset: {str(e)}")
            self.z_offset = None

    def _save_z_offset(self) -> None:
        try:
            with open(self.config.z_offset_file, 'w') as f:
                json.dump({'z_offset': self.z_offset}, f, indent=2)
            self.logger.info(f"Saved Z offset: {self.z_offset}mm")
            self.send_gcode("G92 Z0")
            time.sleep(0.001)
        except Exception as e:
            self.logger.error(f"Failed to save Z offset: {str(e)}")

    def calibrate_position(self, square: str) -> None:
        try:
            existing_pos = self.chess_mapper.get_position(square)
            if existing_pos:
                self.logger.info(f"Current position for {square}:")
                self.logger.info(f"X: {existing_pos['x']:.1f}, Y: {existing_pos['y']:.1f}, Z: {existing_pos['z']:.1f}")
            while True:
                print("\nUse X/Y/Z commands to move to desired position")
                print("Enter 'done' when position is set, or 'cancel' to abort")
                user_input = input("Enter command: ").strip().lower()
                if user_input == 'done':
                    pos = self.get_position()
                    if pos:
                        self.chess_mapper.set_position(square, pos['X'], pos['Y'], pos['Z'])
                        self.logger.info(f"Position saved for {square}")
                    break
                elif user_input == 'cancel':
                    self.logger.info("Calibration cancelled")
                    break
                try:
                    parts = user_input.split()
                    if len(parts) % 2 != 0:
                        raise ValueError("Invalid input format")
                    movements = []
                    for i in range(0, len(parts), 2):
                        axis = parts[i]
                        value = float(parts[i + 1])
                        if axis.upper() in ["X", "Y", "Z"]:
                            movements.append((axis.upper(), value))
                        else:
                            self.logger.warning(f"Invalid axis: {axis}")
                            continue
                    if movements:
                        self.send_batched_commands([
                            "G91",
                            f"G1 {' '.join([f'{a}{v}' for a, v in movements])} F{self.config.xy_feedrate}",
                            "G90"
                        ])
                        new_position = self.get_position()
                        if new_position:
                            self.show_status()
                except ValueError:
                    self.logger.error("Invalid input! Use 'X value Y value Z value' (e.g., 'X 50 Y 30 Z -10')")
        except Exception as e:
            self.logger.error(f"Failed to calibrate position for square {square}: {str(e)}")

    def calibrate_piece(self, piece: str) -> None:
        try:
            existing_settings = self.piece_mapper.get_piece_settings(piece)
            if existing_settings:
                self.logger.info(f"Current settings for {piece}:")
                self.logger.info(f"Height: {existing_settings['height']:.1f}mm")
                self.logger.info(f"Grip: {existing_settings['grip_pw']} pw")
            print("\nOpening gripper...")
            self.open_gripper()
            print("\nStep 1: Height Calibration")
            print("Place the piece under the gripper")
            print("Use Z commands to position gripper just above piece")
            print("Enter 'done' when height is set, or 'cancel' to abort")
            height = None
            while True:
                user_input = input("Enter command: ").strip().lower()
                if user_input == 'done':
                    pos = self.get_position()
                    if pos:
                        height = pos['Z']
                        self.logger.info(f"Height set to {height:.1f}mm")
                    break
                elif user_input == 'cancel':
                    self.logger.info("Calibration cancelled")
                    return
                try:
                    if not user_input.startswith('z '):
                        self.logger.error("Only Z-axis movement allowed for height calibration")
                        continue
                    _, value = user_input.split()
                    value = float(value)
                    self.move_nozzle_smooth([('Z', value)])
                except ValueError:
                    self.logger.error("Invalid input! Use 'Z value', 'done', or 'cancel'")
            if height is not None:
                print("\nStep 2: Grip Calibration")
                print("Gripper is already open")
                print(f"Current range: {self.config.gripper_open_pw} (open) to {self.config.gripper_closed_pw} (closed)")
                print("Enter pulse width value to test grip (e.g., 2400)")
                print("Enter 'done' when grip is set, or 'cancel' to abort")
                grip_pw = None
                while True:
                    user_input = input("Enter command: ").strip().lower()
                    if user_input == 'done' and grip_pw is not None:
                        self.logger.info(f"Grip set to {grip_pw} pw")
                        self.piece_mapper.set_piece_settings(piece, height, grip_pw)
                        self.logger.info(f"Settings saved for {piece}")
                        print("Opening gripper for next piece...")
                        self.open_gripper()
                        break
                    elif user_input == 'cancel':
                        self.logger.info("Calibration cancelled")
                        print("Opening gripper...")
                        self.open_gripper()
                        return
                    try:
                        value = int(user_input)
                        if self.config.gripper_open_pw <= value <= self.config.gripper_closed_pw:
                            grip_pw = value
                            self.pi.set_servo_pulsewidth(self.config.servo_pin, grip_pw)
                        else:
                            self.logger.error(f"Please enter a value between {self.config.gripper_open_pw} and {self.config.gripper_closed_pw}")
                    except ValueError:
                        self.logger.error(f"Invalid input! Use values {self.config.gripper_open_pw}-{self.config.gripper_closed_pw}, 'done', or 'cancel'")
        except Exception as e:
            self.logger.error(f"Failed to calibrate {piece}: {str(e)}")

    def calibrate_storage(self, location: str) -> None:
        try:
            existing_pos = self.storage_mapper.get_position(location)
            if existing_pos:
                self.logger.info(f"Current position for {location}:")
                self.logger.info(f"X: {existing_pos['x']:.1f}, Y: {existing_pos['y']:.1f}, Z: {existing_pos['z']:.1f}")
            while True:
                print("\nUse X/Y/Z commands to move to desired position")
                print("Enter 'done' when position is set, or 'cancel' to abort")
                user_input = input("Enter command: ").strip().lower()
                if user_input == 'done':
                    pos = self.get_position()
                    if pos:
                        self.storage_mapper.set_position(location, pos['X'], pos['Y'], pos['Z'])
                        self.logger.info(f"Position saved for {location}")
                    break
                elif user_input == 'cancel':
                    self.logger.info("Calibration cancelled")
                    break
                try:
                    parts = user_input.split()
                    if len(parts) % 2 != 0:
                        raise ValueError("Invalid input format")
                    movements = []
                    for i in range(0, len(parts), 2):
                        axis = parts[i]
                        value = float(parts[i + 1])
                        if axis.upper() in ["X", "Y", "Z"]:
                            movements.append((axis.upper(), value))
                        else:
                            self.logger.warning(f"Invalid axis: {axis}")
                    if movements:
                        self.send_batched_commands([
                            "G91",
                            f"G1 {' '.join([f'{a}{v}' for a, v in movements])} F{self.config.xy_feedrate}",
                            "G90"
                        ])
                        new_position = self.get_position()
                        if new_position:
                            self.show_status()
                except ValueError:
                    self.logger.error("Invalid input! Use 'X value Y value Z value' (e.g., 'X 50 Y 30 Z -10')")
        except Exception as e:
            self.logger.error(f"Failed to calibrate storage position for {location}: {str(e)}")

    def _ensure_z_reference(self) -> None:
        try:
            self.send_gcode("G90")
            time.sleep(0.001)
            pos = self.get_position()
            if pos and abs(pos['Z']) > 100.0:
                self.logger.warning(f"Unusual Z position detected: {pos['Z']}mm")
        except Exception as e:
            self.logger.error(f"Error checking Z reference: {str(e)}")

    def calculate_z_movement(self, current_z: float, target_z: float, is_up: bool = True) -> float:
        if is_up:
            return abs(target_z - current_z)
        return -(abs(current_z - target_z))



def main():
    config = PrinterConfig()
    chess_config = ChessBoardConfig()
    chess_piece_config = ChessPieceConfig()
    gripper_config = GripperConfig()
    storage_config = StorageConfig()
    controller = None
    
    try:
        # Add debug output
        print("Debug: Initializing main function")
        print(f"Debug: Using printer port: {config.printer_port}")
        
        # Check if pigpio daemon is running
        try:
            print("Debug: Checking pigpio daemon...")
            pi = pigpio.pi()
            if not pi.connected:
                print("Error: pigpio daemon is not running. Please start it with:")
                print("  sudo pigpiod")
                print("or")
                print("  sudo systemctl start pigpiod")
                return
            pi.stop()
            print("Debug: pigpio daemon check successful")
        except Exception as e:
            print(f"Error checking pigpio daemon: {str(e)}")
            print("Full error details:")
            traceback.print_exc()
            print("\nPlease ensure pigpio is installed and the daemon is running with:")
            print("  sudo pigpiod")
            return
            
        print("Debug: Initializing PrinterController...")
        controller = PrinterController(config, chess_config, chess_piece_config, gripper_config, storage_config)
        print("Debug: PrinterController initialized successfully")
        
        # Main control loop
        while True:
            print("\nOptions:")
            print("\nMovement Controls:")
            print("  'X/Y/Z value'        - Move axis (e.g., 'X 50' or 'X 20 Y 30 Z 50')")
            print("  'home'              - Initialize all axes")
            print("  'status'            - Show current positions")
            
            print("\nChess Controls:")
            print("  'play move'         - Execute a chess move (e.g., e4, Nf3, exd5)")
            print("  'play stockfish'    - Play a game against Stockfish")  # Added Stockfish option
            print("  'move to square'    - Move to chess square (a1-h8)")
            
            print("\nCalibration:")
            print("  'calibrate squares' - Calibrate chess square positions")
            print("  'calibrate pieces'  - Calibrate chess piece heights and grips")
            print("  'calibrate storage' - Calibrate storage box positions")
            print("  'calibrate z'       - Recalibrate Z=0 position")
            
            print("\nStorage and Gripper:")
            print("  'move to storage'   - Move to storage box position")
            print("  'gripper'           - Control gripper settings and position")
            
            print("\nSystem:")
            print("  'test'              - Test position commands")
            print("  'done'              - Exit program")
            
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input == 'done':
                controller.logger.info("Session finished")
                break
                
            if user_input == 'home':
                controller.home_axes()
                continue
                
            if user_input == 'status':
                controller.show_status()
                continue
                
            if user_input == 'test':
                controller.test_position_commands()
                continue
                
            if user_input == 'gripper':
                controller.control_gripper()
                continue
                
            if user_input == 'calibrate z':  
                controller.recalibrate_z_offset()
                continue
                
            if user_input == 'play stockfish':  # Added Stockfish handler
                controller.play_against_stockfish()
                continue
                
            if user_input == 'play move':
                while True:
                    print("\nChess Move Mode")
                    print("Enter move in algebraic notation:")
                    print("  - Pawn moves: e4, d5")
                    print("  - Piece moves: Nf3, Bd3")
                    print("  - Captures: exd5, Bxe4")
                    print("  - Castling: O-O, O-O-O")
                    print("Or 'done' to return to main menu")
                    
                    move = input("Move: ").strip()
                    
                    if move.lower() == 'done':
                        break
                        
                    if not move:
                        print("Please enter a move")
                        continue
                        
                    controller.play_move(move)
                continue
                
            if user_input == 'move to storage':
                while True:
                    print("\nEnter storage location (box_1/box_2)....box 1 is the box closer to you")
                    print("Or 'done' to return to main menu")
                    
                    target = input("Location: ").strip().lower()
                    
                    if target == 'done':
                        break
                        
                    if target not in storage_config.positions:
                        print(f"Invalid location! Choose from: {', '.join(storage_config.positions)}")
                        continue
                        
                    controller.move_to_storage(target)
                continue
                
            if user_input == 'calibrate storage':
                while True:
                    print("\nStorage Calibration Mode")
                    print("Available locations:", ", ".join(storage_config.positions))
                    print("Enter location to calibrate")
                    print("Or 'done' to return to main menu")
                    
                    location = input("Location: ").strip().lower()
                    
                    if location == 'done':
                        break
                        
                    if location not in storage_config.positions:
                        print(f"Invalid location! Choose from: {', '.join(storage_config.positions)}")
                        continue
                        
                    controller.calibrate_storage(location)
                continue
                
            if user_input == 'move to square':
                while True:
                    print("\nEnter chess square (a1-h8)")
                    print("Or 'done' to return to main menu")
                    
                    target = input("Square: ").strip().lower()
                    
                    if target == 'done':
                        break
                        
                    if not re.match(r'^[a-h][1-8]$', target):
                        controller.logger.error("Invalid square! Use a1-h8")
                        continue
                        
                    controller.move_to_square_smooth(target)
                continue
                
            if user_input == 'calibrate squares':
                while True:
                    print("\nSquares Calibration Mode")
                    print("Enter chess square to calibrate (a1-h8)")
                    print("Or 'done' to return to main menu")
                    
                    square = input("Square: ").strip().lower()
                    
                    if square == 'done':
                        break
                        
                    if not re.match(r'^[a-h][1-8]$', square):
                        print("Invalid chess square! Use a1-h8")
                        continue
                        
                    controller.calibrate_position(square)
                continue
                
            if user_input == 'calibrate pieces':
                while True:
                    print("\nPiece Calibration Mode")
                    print("Available pieces:", ", ".join(chess_piece_config.pieces))
                    print("Enter piece name to calibrate")
                    print("Or 'done' to return to main menu")
                    
                    piece = input("Piece: ").strip().lower()
                    
                    if piece == 'done':
                        break
                        
                    if piece not in chess_piece_config.pieces:
                        print(f"Invalid piece! Choose from: {', '.join(chess_piece_config.pieces)}")
                        continue
                        
                    controller.calibrate_piece(piece)
                continue
                
            # Handle direct axis movements
            try:
                # Split input into parts and check if they come in pairs
                parts = user_input.split()
                if len(parts) % 2 != 0:
                    raise ValueError("Invalid input format")
                
                # Process each axis-value pair
                movements = []
                for i in range(0, len(parts), 2):
                    axis = parts[i]
                    value = float(parts[i + 1])
                    
                    if axis.upper() in ["X", "Y", "Z"]:
                        movements.append((axis.upper(), value))
                    else:
                        controller.logger.warning(f"Invalid axis: {axis}")
                        continue
                
                # Execute movements
                if movements:
                    # Send a single G-code command for all movements
                    controller.send_gcode("G91")  # Relative positioning
                    movement_str = " ".join([f"{axis}{value}" for axis, value in movements])
                    controller.send_gcode(f"G1 {movement_str} F{controller.config.xy_feedrate}")
                    controller.send_gcode("G90")  # Back to absolute
                    
                    # Get and show new position
                    new_position = controller.get_position()
                    if new_position:
                        controller.show_status()
                        
                        # Ask about gripper control
                        action = input(
                            f"Do you want to {'CLOSE' if controller.gripper_state == 'open' else 'OPEN'} "
                            f"the gripper? (yes/no): "
                        ).strip().lower()
                        
                        if action == "yes":
                            if controller.gripper_state == "open":
                                controller.close_gripper()
                            else:
                                controller.open_gripper()
                    
            except ValueError:
                controller.logger.error("Invalid input! Format: 'X value Y value Z value' (e.g., 'X 50 Y 30 Z -10')")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"\nFatal error during initialization: {str(e)}")
        print("Full error details:")
        traceback.print_exc()
    finally:
        if controller:
            try:
                controller.cleanup()
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")

if __name__ == "__main__":
    try:
        print("Starting printer control program...")
        print("Debug: Python version:", sys.version)
        print("Debug: Current working directory:", os.getcwd())
        
        # Check for required files
        required_files = [
            PrinterConfig().z_offset_file,
            ChessBoardConfig().positions_file,
            ChessPieceConfig().pieces_file,
            GripperConfig().settings_file,
            StorageConfig().storage_file
        ]
        
        print("\nDebug: Checking required files:")
        for file in required_files:
            exists = os.path.exists(file)
            print(f"  {file}: {'Found' if exists else 'Missing'}")
        
        # Check if serial port exists
        port = PrinterConfig().printer_port
        print(f"\nDebug: Checking serial port {port}")
        if os.path.exists(port):
            print(f"Serial port {port} exists")
        else:
            print(f"Warning: Serial port {port} not found")
            
        main()
        
    except Exception as e:
        print(f"\nStartup error: {str(e)}")
        print("Full error details:")
        traceback.print_exc()
    finally:
        print("\nPress return to continue...")
        input()  # Wait for return
        sys.exit(0)  # Exit the program


