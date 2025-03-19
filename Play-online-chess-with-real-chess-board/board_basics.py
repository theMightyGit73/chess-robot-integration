import sys
import os
import pickle
import chess
import logging
import numpy as np
from typing import List, Tuple, Set, Dict, Any, Optional
from skimage.metrics import structural_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BoardBasics")

class Board_basics:
    """
    Provides fundamental chess board analysis functionality including:
    - Square detection and naming
    - Image similarity analysis for piece detection
    - Move detection based on image changes
    """
    
    def __init__(self, side_view_compensation: Tuple[int, int], rotation_count: int):
        """
        Initialize the board basics with calibration parameters.
        
        Args:
            side_view_compensation: Tuple indicating pixel offset for side view correction
            rotation_count: Number of 90-degree rotations to apply (0-3)
        """
        self.d = [side_view_compensation, (0, 0)]  # Direction vectors for region calculation
        self.rotation_count = rotation_count % 4  # Ensure rotation is in valid range
        
        # Default SSIM thresholds for different square types
        self.SSIM_THRESHOLD = 0.8  # General threshold
        self.SSIM_THRESHOLD_LIGHT_WHITE = 1.0  # Light square with white piece
        self.SSIM_THRESHOLD_LIGHT_BLACK = 1.0  # Light square with black piece
        self.SSIM_THRESHOLD_DARK_WHITE = 1.0   # Dark square with white piece
        self.SSIM_THRESHOLD_DARK_BLACK = 1.0   # Dark square with black piece
        
        # Organized threshold table: [dark/light][black/white]
        self.ssim_table = [
            [self.SSIM_THRESHOLD_DARK_BLACK, self.SSIM_THRESHOLD_DARK_WHITE],
            [self.SSIM_THRESHOLD_LIGHT_BLACK, self.SSIM_THRESHOLD_LIGHT_WHITE]
        ]
        
        # Save file for SSIM parameters
        self.save_file = "ssim.bin"
        
        # Create config directory if needed
        config_dir = os.path.join(os.path.expanduser("~"), ".chess_robot")
        os.makedirs(config_dir, exist_ok=True)
        self.save_file = os.path.join(config_dir, self.save_file)
        
        logger.info(f"Board basics initialized with rotation_count={rotation_count}")
        logger.debug(f"Side view compensation: {side_view_compensation}")

    def initialize_ssim(self, frame: np.ndarray) -> None:
        """
        Initialize structural similarity (SSIM) thresholds based on the current board state.
        This is typically done at the start of a game with pieces in their initial positions.
        
        Args:
            frame: Image of the chess board
            
        Raises:
            ValueError: If frame is invalid or empty
        """
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame provided for SSIM initialization")
                
            logger.info("Initializing SSIM thresholds")
            
            # Collect square images for analysis
            light_white = []
            dark_white = []
            light_empty = []
            dark_empty = []
            light_black = []
            dark_black = []
            
            # Iterate through all squares
            for row in range(8):
                for column in range(8):
                    square_name = self.convert_row_column_to_square_name(row, column)
                    
                    try:
                        square_img = self.get_square_image(row, column, frame)
                        
                        # Categorize squares based on their positions and colors
                        if square_name[1] == "2":  # White pawns
                            if self.is_light(square_name):
                                light_white.append(square_img)
                            else:
                                dark_white.append(square_img)
                        elif square_name[1] == "4":  # Empty squares
                            if self.is_light(square_name):
                                light_empty.append(square_img)
                            else:
                                dark_empty.append(square_img)
                        elif square_name[1] == "7":  # Black pawns
                            if self.is_light(square_name):
                                light_black.append(square_img)
                            else:
                                dark_black.append(square_img)
                    except Exception as square_error:
                        logger.warning(f"Error processing square {square_name}: {square_error}")
            
            # Calculate similarity scores
            try:
                # Light squares with white pieces
                ssim_light_white = max(
                    structural_similarity(empty, piece, channel_axis=-1) 
                    for piece, empty in zip(light_white, light_empty)
                ) if light_white and light_empty else 0.8
                
                # Light squares with black pieces
                ssim_light_black = max(
                    structural_similarity(empty, piece, channel_axis=-1) 
                    for piece, empty in zip(light_black, light_empty)
                ) if light_black and light_empty else 0.8
                
                # Dark squares with white pieces
                ssim_dark_white = max(
                    structural_similarity(empty, piece, channel_axis=-1) 
                    for piece, empty in zip(dark_white, dark_empty)
                ) if dark_white and dark_empty else 0.8
                
                # Dark squares with black pieces
                ssim_dark_black = max(
                    structural_similarity(empty, piece, channel_axis=-1) 
                    for piece, empty in zip(dark_black, dark_empty)
                ) if dark_black and dark_empty else 0.8
                
                # Update thresholds
                self.SSIM_THRESHOLD_LIGHT_WHITE = min(self.SSIM_THRESHOLD_LIGHT_WHITE, ssim_light_white + 0.2)
                self.SSIM_THRESHOLD_LIGHT_BLACK = min(self.SSIM_THRESHOLD_LIGHT_BLACK, ssim_light_black + 0.2)
                self.SSIM_THRESHOLD_DARK_WHITE = min(self.SSIM_THRESHOLD_DARK_WHITE, ssim_dark_white + 0.2)
                self.SSIM_THRESHOLD_DARK_BLACK = min(self.SSIM_THRESHOLD_DARK_BLACK, ssim_dark_black + 0.2)
                
                # Set overall threshold to highest individual threshold
                self.SSIM_THRESHOLD = max([
                    self.SSIM_THRESHOLD, 
                    self.SSIM_THRESHOLD_LIGHT_WHITE, 
                    self.SSIM_THRESHOLD_LIGHT_BLACK,
                    self.SSIM_THRESHOLD_DARK_WHITE, 
                    self.SSIM_THRESHOLD_DARK_BLACK
                ])
                
                # Update threshold table
                self.ssim_table = [
                    [self.SSIM_THRESHOLD_DARK_BLACK, self.SSIM_THRESHOLD_DARK_WHITE],
                    [self.SSIM_THRESHOLD_LIGHT_BLACK, self.SSIM_THRESHOLD_LIGHT_WHITE]
                ]
                
                logger.info(f"SSIM thresholds: LW={self.SSIM_THRESHOLD_LIGHT_WHITE:.3f}, LB={self.SSIM_THRESHOLD_LIGHT_BLACK:.3f}, " + 
                           f"DW={self.SSIM_THRESHOLD_DARK_WHITE:.3f}, DB={self.SSIM_THRESHOLD_DARK_BLACK:.3f}")
                
                # Save thresholds
                self.save_ssim_thresholds()
                
            except Exception as e:
                logger.error(f"Error calculating SSIM thresholds: {e}")
                
        except Exception as e:
            logger.error(f"SSIM initialization error: {e}")
            raise
    
    def save_ssim_thresholds(self) -> bool:
        """
        Save SSIM thresholds to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.save_file, 'wb') as outfile:
                thresholds = (
                    self.SSIM_THRESHOLD_LIGHT_WHITE,
                    self.SSIM_THRESHOLD_LIGHT_BLACK,
                    self.SSIM_THRESHOLD_DARK_WHITE,
                    self.SSIM_THRESHOLD_DARK_BLACK,
                    self.SSIM_THRESHOLD
                )
                pickle.dump(thresholds, outfile)
            logger.info(f"SSIM thresholds saved to {self.save_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving SSIM thresholds: {e}")
            return False

    def load_ssim(self) -> bool:
        """
        Load SSIM thresholds from file.
        
        Returns:
            True if successful, False if file not found or error
        """
        try:
            if not os.path.exists(self.save_file):
                logger.error(f"SSIM file not found: {self.save_file}")
                return False
                
            with open(self.save_file, 'rb') as infile:
                (
                    self.SSIM_THRESHOLD_LIGHT_WHITE,
                    self.SSIM_THRESHOLD_LIGHT_BLACK, 
                    self.SSIM_THRESHOLD_DARK_WHITE,
                    self.SSIM_THRESHOLD_DARK_BLACK, 
                    self.SSIM_THRESHOLD
                ) = pickle.load(infile)
                
            # Update threshold table
            self.ssim_table = [
                [self.SSIM_THRESHOLD_DARK_BLACK, self.SSIM_THRESHOLD_DARK_WHITE],
                [self.SSIM_THRESHOLD_LIGHT_BLACK, self.SSIM_THRESHOLD_LIGHT_WHITE]
            ]
            
            logger.info(f"SSIM thresholds loaded: LW={self.SSIM_THRESHOLD_LIGHT_WHITE:.3f}, " + 
                       f"LB={self.SSIM_THRESHOLD_LIGHT_BLACK:.3f}, " + 
                       f"DW={self.SSIM_THRESHOLD_DARK_WHITE:.3f}, " + 
                       f"DB={self.SSIM_THRESHOLD_DARK_BLACK:.3f}")
            return True
            
        except FileNotFoundError:
            logger.error("SSIM file not found. You need to play at least 1 game before starting a game from position.")
            return False
        except Exception as e:
            logger.error(f"Error loading SSIM thresholds: {e}")
            return False

    def update_ssim(self, previous_frame: np.ndarray, next_frame: np.ndarray, 
                   move: chess.Move, is_capture: bool, color: int) -> None:
        """
        Update SSIM thresholds based on a detected move.
        
        Args:
            previous_frame: The board image before the move
            next_frame: The board image after the move
            move: The chess move that was made
            is_capture: Whether the move was a capture
            color: The color of the moving piece (0=black, 1=white)
            
        Raises:
            ValueError: If frames or move are invalid
        """
        try:
            # Get square names
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            
            # Check each affected square
            for row in range(8):
                for column in range(8):
                    square_name = self.convert_row_column_to_square_name(row, column)
                    
                    # Skip squares not involved in the move
                    if square_name not in [from_square, to_square]:
                        continue
                        
                    # Get square images before and after move
                    try:
                        previous_square = self.get_square_image(row, column, previous_frame)
                        next_square = self.get_square_image(row, column, next_frame)
                        
                        # Calculate similarity
                        ssim = structural_similarity(next_square, previous_square, channel_axis=-1)
                        ssim += 0.1  # Add margin
                        
                        # Update global threshold if needed
                        if ssim > self.SSIM_THRESHOLD:
                            self.SSIM_THRESHOLD = ssim
                            logger.debug(f"Updated global SSIM threshold to {ssim:.3f}")
                        
                        # Update specific threshold for this square type
                        is_light = int(self.is_light(square_name))
                        
                        # Only update if this is the origin square or not a capture
                        if (square_name == from_square) or (not is_capture):
                            if ssim > self.ssim_table[is_light][color]:
                                self.ssim_table[is_light][color] = ssim
                                logger.debug(f"Updated SSIM threshold for {'light' if is_light else 'dark'} " + 
                                           f"square with {'white' if color else 'black'} piece to {ssim:.3f}")
                    
                    except Exception as square_error:
                        logger.warning(f"Error updating SSIM for square {square_name}: {square_error}")
            
            # Save updated thresholds
            self.save_ssim_thresholds()
            
        except Exception as e:
            logger.error(f"Error updating SSIM thresholds: {e}")

    def get_square_image(self, row: int, column: int, board_img: np.ndarray) -> np.ndarray:
        """
        Extract an image of a specific square from the board image.
        
        Args:
            row: Row index (0-7)
            column: Column index (0-7)
            board_img: Full board image
            
        Returns:
            Image of the specified square
            
        Raises:
            ValueError: If row/column is invalid or board_img is None
        """
        try:
            # Input validation
            if not isinstance(board_img, np.ndarray) or board_img.size == 0:
                raise ValueError("Invalid board image")
                
            if not (0 <= row < 8) and not (0 <= column < 8):
                raise ValueError(f"Invalid row/column: {row}, {column}")
            
            # Get image dimensions
            height, width = board_img.shape[:2]
            
            # Calculate square boundaries
            min_x = int(column * width / 8)
            max_x = int((column + 1) * width / 8)
            min_y = int(row * height / 8)
            max_y = int((row + 1) * height / 8)
            
            # Extract and return the square
            return board_img[min_y:max_y, min_x:max_x]
            
        except Exception as e:
            logger.error(f"Error extracting square image at ({row}, {column}): {e}")
            # Return a small blank image as fallback
            return np.zeros((10, 10, 3) if len(board_img.shape) == 3 else (10, 10), dtype=np.uint8)

    def convert_row_column_to_square_name(self, row: int, column: int) -> str:
        """
        Convert row and column indices to chess square name, respecting board rotation.
        
        Args:
            row: Row index (0-7)
            column: Column index (0-7)
            
        Returns:
            Chess square name (e.g., 'e4')
            
        Raises:
            ValueError: If row/column is invalid
        """
        try:
            # Validate inputs
            if not (0 <= row < 8) or not (0 <= column < 8):
                raise ValueError(f"Invalid row/column: {row}, {column}")
            
            # Apply rotation to get correct square name
            if self.rotation_count == 0:
                # Standard orientation (a1 at bottom left)
                number = str(8 - row)
                letter = chr(97 + column)
            elif self.rotation_count == 1:
                # Rotated 90° clockwise
                number = str(8 - column)
                letter = chr(97 + (7 - row))
            elif self.rotation_count == 2:
                # Rotated 180°
                number = str(row + 1)
                letter = chr(97 + (7 - column))
            elif self.rotation_count == 3:
                # Rotated 270° clockwise
                number = str(column + 1)
                letter = chr(97 + row)
            else:
                # Should never happen due to modulo in __init__
                raise ValueError(f"Invalid rotation count: {self.rotation_count}")
                
            return letter + number
            
        except Exception as e:
            logger.error(f"Error converting row/column to square name: {e}")
            return "a1"  # Default fallback

    def convert_square_name_to_row_column(self, square_name):
        """
        Convert chess notation (e.g., 'e4') to row and column indices (0-7)
        
        Args:
            square_name: String in chess notation (e.g., 'e4')
            
        Returns:
            tuple: (row, col) where both are 0-7 indices
        """
        try:
            if not square_name or len(square_name) != 2:
                return None, None
                
            file_char = square_name[0].lower()
            rank_char = square_name[1]
            
            if not ('a' <= file_char <= 'h') or not ('1' <= rank_char <= '8'):
                return None, None
            
            # Convert file (column) and rank (row) to 0-7 indices
            col = ord(file_char) - ord('a')
            row = 8 - int(rank_char)
            
            # Apply board rotation if needed
            if self.rotation_count == 1:  # 90 degrees
                old_row, old_col = row, col
                row, col = old_col, 7 - old_row
            elif self.rotation_count == 2:  # 180 degrees
                row, col = 7 - row, 7 - col
            elif self.rotation_count == 3:  # 270 degrees
                old_row, old_col = row, col
                row, col = 7 - old_col, old_row
                
            return row, col
            
        except Exception as e:
            logger.error(f"Error converting square name to row/column: {e}")
            return None, None

    def square_region(self, row: int, column: int) -> Set[Tuple[int, int]]:
        """
        Calculate the region of squares affected by side view compensation.
        
        Args:
            row: Row index (0-7)
            column: Column index (0-7)
            
        Returns:
            Set of (row, column) tuples representing affected squares
        """
        try:
            region = set()
            
            # Apply each direction vector
            for d_row, d_column in self.d:
                n_row = row + d_row
                n_column = column + d_column
                
                # Skip if out of bounds
                if not (0 <= n_row < 8) or not (0 <= n_column < 8):
                    continue
                    
                region.add((n_row, column))
                
            return region
            
        except Exception as e:
            logger.error(f"Error calculating square region: {e}")
            return {(row, column)}  # Default to just the original square

    def is_light(self, square_name: str) -> bool:
        """
        Determine if a square is light-colored or dark-colored.
        
        Args:
            square_name: Chess square name (e.g., 'e4')
            
        Returns:
            True if the square is light-colored, False if dark-colored
            
        Raises:
            ValueError: If square_name is invalid
        """
        try:
            if not isinstance(square_name, str) or len(square_name) != 2:
                raise ValueError(f"Invalid square name: {square_name}")
                
            file_char = square_name[0].lower()
            rank_char = square_name[1]
            
            if not ('a' <= file_char <= 'h') or not ('1' <= rank_char <= '8'):
                raise ValueError(f"Invalid square name: {square_name}")
            
            # Light squares are those where file and rank have the same parity
            if file_char in "aceg":
                return square_name[1] in "2468"
            else:
                return square_name[1] in "1357"
                
        except Exception as e:
            logger.error(f"Error determining square color for {square_name}: {e}")
            return False  # Default fallback

    def get_potential_moves(self, fgmask: np.ndarray, previous_frame: np.ndarray, 
                           next_frame: np.ndarray, chessboard: chess.Board) -> Tuple[List, List]:
        """
        Detect potential chess moves based on changes between two frames.
        
        Args:
            fgmask: Foreground mask highlighting changed areas
            previous_frame: Frame before the move
            next_frame: Frame after the move
            chessboard: Current chess board state
            
        Returns:
            Tuple of (potential_squares_for_castling, potential_moves)
            
        Raises:
            ValueError: If any input is invalid
        """
        try:
            # Validate inputs
            if not isinstance(chessboard, chess.Board):
                raise ValueError("Invalid chessboard object")
                
            for frame in [fgmask, previous_frame, next_frame]:
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise ValueError("Invalid frame")
            
            # Calculate mean values for foreground mask
            board = [[self.get_square_image(row, column, fgmask).mean() for column in range(8)] for row in range(8)]
            
            # Get square images for previous and next frames
            previous_board = [[self.get_square_image(row, column, previous_frame) for column in range(8)] for row in range(8)]
            next_board = [[self.get_square_image(row, column, next_frame) for column in range(8)] for row in range(8)]
            
            # Detect changed squares
            potential_squares = []
            for row in range(8):
                for column in range(8):
                    # Get score from foreground mask
                    score = board[row][column]
                    
                    # Skip squares with low change
                    if score < 10.0:
                        continue

                    # Calculate similarity between previous and current square
                    try:
                        ssim = structural_similarity(
                            next_board[row][column],
                            previous_board[row][column], 
                            channel_axis=-1
                        )
                        
                        square_name = self.convert_row_column_to_square_name(row, column)
                        logger.debug(f"SSIM={ssim:.3f} for square {square_name}")
                        
                        # Skip if similarity is above threshold (no significant change)
                        if ssim > self.SSIM_THRESHOLD:
                            continue
                            
                        # Get chess piece at this square
                        square = chess.parse_square(square_name)
                        piece = chessboard.piece_at(square)
                        
                        # Additional check for pieces of the current turn
                        if piece and piece.color == chessboard.turn:
                            is_light = int(self.is_light(square_name))
                            color = int(piece.color)
                            
                            # Skip if similarity is above piece-specific threshold
                            if ssim > self.ssim_table[is_light][color]:
                                continue
                                
                        # Add to potential squares
                        potential_squares.append((score, row, column, ssim))
                        
                    except Exception as square_error:
                        logger.warning(f"Error processing square ({row}, {column}): {square_error}")

            # Sort by score (descending)
            potential_squares.sort(reverse=True)
            
            # Prepare potential squares for castling detection
            potential_squares_castling = []
            for i in range(min(6, len(potential_squares))):
                score, row, column, ssim = potential_squares[i]
                square_name = self.convert_row_column_to_square_name(row, column)
                potential_squares_castling.append((score, square_name))
            
            # Limit to top candidates for move analysis
            potential_squares = potential_squares[:4]
            potential_moves = []

            # Generate potential moves by combining start and arrival squares
            for start_square_score, start_row, start_column, start_ssim in potential_squares:
                # Get start square info
                start_square_name = self.convert_row_column_to_square_name(start_row, start_column)
                start_square = chess.parse_square(start_square_name)
                start_piece = chessboard.piece_at(start_square)
                
                # Skip if no piece or wrong color
                if not start_piece or start_piece.color != chessboard.turn:
                    continue
                    
                # Calculate affected region around start square
                start_region = self.square_region(start_row, start_column)
                
                # Try each potential arrival square
                for arrival_square_score, arrival_row, arrival_column, arrival_ssim in potential_squares:
                    # Skip if same square
                    if (start_row, start_column) == (arrival_row, arrival_column):
                        continue
                        
                    # Get arrival square info
                    arrival_square_name = self.convert_row_column_to_square_name(arrival_row, arrival_column)
                    arrival_square = chess.parse_square(arrival_square_name)
                    arrival_piece = chessboard.piece_at(arrival_square)
                    
                    # Skip if arrival square has our piece
                    if arrival_piece and arrival_piece.color == chessboard.turn:
                        continue
                        
                    # Additional check for empty squares
                    if not arrival_piece:
                        is_light = int(self.is_light(arrival_square_name))
                        color = int(start_piece.color)
                        
                        # Skip if similarity is above threshold
                        if arrival_ssim > self.ssim_table[is_light][color]:
                            continue
                    
                    # Calculate combined region
                    arrival_region = self.square_region(arrival_row, arrival_column)
                    region = start_region.union(arrival_region)
                    
                    # Calculate total score for this potential move
                    total_square_score = sum(
                        board[row][column] for row, column in region
                    ) + start_square_score + arrival_square_score
                    
                    # Add to potential moves
                    potential_moves.append(
                        (total_square_score, start_square_name, arrival_square_name)
                    )

            # Sort by score (descending)
            potential_moves.sort(reverse=True)
            
            return potential_squares_castling, potential_moves
            
        except Exception as e:
            logger.error(f"Error detecting potential moves: {e}")
            return [], []


# Test function
def test_board_basics():
    """Test the Board_basics class with a simple example."""
    try:
        print("Testing Board_basics class...")
        
        # Create a basic board
        board = Board_basics((0, 0), 0)
        
        # Test square name conversion
        for row in range(8):
            for col in range(8):
                square = board.convert_row_column_to_square_name(row, col)
                print(f"({row}, {col}) -> {square} ({'light' if board.is_light(square) else 'dark'})")
                
        # Test rotation handling
        for rotation in range(4):
            rotated_board = Board_basics((0, 0), rotation)
            print(f"\nRotation {rotation}:")
            print(f"a1 = {rotated_board.convert_row_column_to_square_name(7, 0)}")
            print(f"h8 = {rotated_board.convert_row_column_to_square_name(0, 7)}")
        
        print("\nBoard_basics tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_board_basics()
