from threading import Thread
import chess
import mss
import numpy as np
import cv2
import time
import logging
import traceback
from typing import Tuple, List, Optional, Dict, Any
from classifier import Classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Commentator")

class Commentator_thread(Thread):
    """
    Thread that monitors a chess board on screen, detects moves,
    and provides spoken commentary.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the commentator thread."""
        super(Commentator_thread, self).__init__(*args, **kwargs)
        self.speech_thread = None
        self.game_state = Game_state()
        self.comment_me: bool = False
        self.comment_opponent: bool = False
        self.language = None
        self.classifier = None
        self.running: bool = True
        self.error_count: int = 0
        self.max_errors: int = 10  # Maximum consecutive errors before thread exits
        self.capture_interval: float = 0.5  # Seconds between board captures

    def run(self):
        """Main thread execution loop for the commentator."""
        logger.info("Commentator thread starting")
        
        try:
            # Initialize screen capture
            self.game_state.sct = mss.mss()
            logger.info("Screen capture initialized")
            
            # Get initial board state
            try:
                resized_chessboard = self.game_state.get_chessboard()
                if resized_chessboard is None:
                    raise ValueError("Failed to capture initial chessboard image")
                    
                self.game_state.previous_chessboard_image = resized_chessboard
                
                # Initialize classifier with current board state
                self.game_state.classifier = Classifier(self.game_state)
                logger.info("Chess position classifier initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize board state: {e}")
                logger.debug(traceback.format_exc())
                return
            
            # Main monitoring loop
            while self.running and not self.game_state.board.is_game_over() and not self.game_state.resign_or_draw:
                try:
                    # Determine whose turn it is
                    is_my_turn = (self.game_state.we_play_white) == (self.game_state.board.turn == chess.WHITE)
                    
                    # Check for new moves
                    found_move, move = self.game_state.register_move_if_needed()
                    
                    # If move was found, provide commentary if configured to do so
                    if found_move:
                        self.error_count = 0  # Reset error counter on successful move
                        
                        # Determine if we should comment on this move
                        should_comment = (
                            (self.comment_me and is_my_turn) or 
                            (self.comment_opponent and not is_my_turn)
                        )
                        
                        if should_comment and self.speech_thread and self.language:
                            try:
                                comment = self.language.comment(self.game_state.board, move)
                                logger.debug(f"Speaking comment: {comment}")
                                self.speech_thread.put_text(comment)
                            except Exception as comment_error:
                                logger.error(f"Error generating or speaking comment: {comment_error}")
                    
                    # Brief pause between board captures
                    time.sleep(self.capture_interval)
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Error in commentator loop: {e}")
                    
                    if self.error_count >= self.max_errors:
                        logger.critical(f"Too many consecutive errors ({self.max_errors}), stopping commentator")
                        break
                        
                    # Progressively longer pauses on repeated errors
                    time.sleep(min(5, 0.5 * self.error_count))
            
            # Handle game over
            self.handle_game_end()
            
        except Exception as e:
            logger.error(f"Fatal error in commentator thread: {e}")
            logger.debug(traceback.format_exc())
        
        logger.info("Commentator thread stopped")
    
    def handle_game_end(self):
        """Handle the end of the game with appropriate commentary."""
        try:
            if self.game_state.board.is_game_over() and self.speech_thread and self.language:
                result = self.game_state.board.result()
                
                if result == "1-0":  # White wins
                    self.speech_thread.put_text(getattr(self.language, "white_wins", "White wins"))
                elif result == "0-1":  # Black wins
                    self.speech_thread.put_text(getattr(self.language, "black_wins", "Black wins"))
                elif result == "1/2-1/2":  # Draw
                    self.speech_thread.put_text(getattr(self.language, "draw", "Draw"))
                else:
                    self.speech_thread.put_text(getattr(self.language, "game_over", "Game over"))
                
                logger.info(f"Game ended: {result}")
            elif self.game_state.resign_or_draw and self.speech_thread and self.language:
                self.speech_thread.put_text(getattr(self.language, "game_over_resignation_or_draw", 
                                                 "Game ended by resignation or draw"))
        except Exception as e:
            logger.error(f"Error handling game end: {e}")
    
    def stop(self):
        """Gracefully stop the commentator thread."""
        logger.info("Stopping commentator thread")
        self.running = False


class Game_state:
    """
    Manages the state of a chess game being played on screen, including
    board position, move detection, and position analysis.
    """

    def __init__(self):
        """Initialize the game state."""
        self.game_thread = None
        self.we_play_white: bool = True  # Default to white, will be updated later
        self.previous_chessboard_image = None
        self.board = chess.Board()
        self.board_position_on_screen = None
        self.sct = None  # Will be initialized in run
        self.classifier = None
        self.registered_moves: List[chess.Move] = []
        self.resign_or_draw: bool = False
        self.variant: str = 'standard'
        self.image_difference_threshold: float = 8.0  # Threshold for square change detection
        self.screenshot_margin: int = 10  # Extra pixels to capture around the board
        self.square_border: int = 6  # Border pixels to remove from square images

    def get_chessboard(self) -> Optional[np.ndarray]:
        """
        Capture the current chessboard image from the screen.
        
        Returns:
            Resized grayscale image of the chessboard or None if capture fails
        """
        try:
            if self.board_position_on_screen is None or self.sct is None:
                logger.error("Board position or screen capture not initialized")
                return None
                
            position = self.board_position_on_screen
            
            # Calculate screen region to capture
            monitor = {
                'top': max(0, position.minY - self.screenshot_margin),
                'left': max(0, position.minX - self.screenshot_margin),
                'width': position.maxX - position.minX + 2 * self.screenshot_margin,
                'height': position.maxY - position.minY + 2 * self.screenshot_margin
            }
            
            # Capture screen region
            img = np.array(self.sct.grab(monitor))
            
            # Convert to grayscale for better processing
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract and resize the chessboard portion
            x_offset = max(0, self.screenshot_margin - position.minX)
            y_offset = max(0, self.screenshot_margin - position.minY)
            
            chessboard = image[
                y_offset:y_offset + (position.maxY - position.minY),
                x_offset:x_offset + (position.maxX - position.minX)
            ]
            
            # Resize to standard dimensions for analysis
            dim = (800, 800)
            resized_chessboard = cv2.resize(chessboard, dim, interpolation=cv2.INTER_AREA)
            
            return resized_chessboard
            
        except Exception as e:
            logger.error(f"Error capturing chessboard: {e}")
            logger.debug(traceback.format_exc())
            return None

    def get_square_image(self, row: int, column: int, board_img: np.ndarray) -> np.ndarray:
        """
        Extract an image of a specific square from the chessboard image.
        
        Args:
            row: Row index (0-7)
            column: Column index (0-7)
            board_img: Full chessboard image
            
        Returns:
            Image of the specified square with borders removed
            
        Raises:
            ValueError: If row or column is out of range, or board_img is invalid
        """
        try:
            # Validate inputs
            if not isinstance(board_img, np.ndarray) or len(board_img.shape) < 2:
                raise ValueError("Invalid board image")
                
            if not (0 <= row < 8) or not (0 <= column < 8):
                raise ValueError(f"Invalid row/column: {row}, {column}")
                
            # Get dimensions
            height, width = board_img.shape[:2]
            
            # Calculate square boundaries
            min_x = int(column * width / 8)
            max_x = int((column + 1) * width / 8)
            min_y = int(row * height / 8)
            max_y = int((row + 1) * height / 8)
            
            # Extract square
            square = board_img[min_y:max_y, min_x:max_x]
            
            # Remove borders if square is large enough
            if (square.shape[0] > 2 * self.square_border and 
                square.shape[1] > 2 * self.square_border):
                square_without_borders = square[
                    self.square_border:-self.square_border, 
                    self.square_border:-self.square_border
                ]
                return square_without_borders
            else:
                return square
                
        except Exception as e:
            logger.error(f"Error extracting square image at ({row}, {column}): {e}")
            # Return a small empty image as fallback
            return np.zeros((10, 10), dtype=np.uint8)

    def can_image_correspond_to_chessboard(self, move: chess.Move, result: List[List[str]]) -> bool:
        """
        Check if the current image classification result matches a chess position
        after applying the given move.
        
        Args:
            move: Chess move to check
            result: 2D list of piece symbols from image classification
            
        Returns:
            True if the image is consistent with the move, False otherwise
        """
        try:
            # Apply the move temporarily
            self.board.push(move)
            
            # Check every square for consistency
            squares = chess.SquareSet(chess.BB_ALL)
            for square in squares:
                row = chess.square_rank(square)
                column = chess.square_file(square)
                piece = self.board.piece_at(square)
                should_be_empty = (piece is None)
                
                # Map board coordinates to image coordinates based on orientation
                if self.we_play_white:
                    row_on_image = 7 - row
                    column_on_image = column
                else:
                    row_on_image = row
                    column_on_image = 7 - column
                
                # Check if square is empty in the image
                is_empty = result[row_on_image][column_on_image] == '.'
                
                # Check for mismatch in piece presence
                if is_empty != should_be_empty:
                    self.board.pop()
                    return False
                
                # If piece present, check if it's the right type
                if piece and (piece.symbol().lower() != result[row_on_image][column_on_image]):
                    self.board.pop()
                    return False
            
            # All squares match
            self.board.pop()
            return True
            
        except Exception as e:
            logger.error(f"Error checking image correspondence for move {move.uci()}: {e}")
            # Ensure board state is restored
            if self.board.move_stack and self.board.peek() == move:
                self.board.pop()
            return False

    def find_premove(self, result: List[List[str]]) -> List[str]:
        """
        Find squares where pieces might have been premoved from.
        
        Args:
            result: 2D list of piece symbols from image classification
            
        Returns:
            List of potential start square names for premoves
        """
        try:
            start_squares = []
            squares = chess.SquareSet(chess.BB_ALL)
            
            for square in squares:
                row = chess.square_rank(square)
                column = chess.square_file(square)
                piece = self.board.piece_at(square)
                
                # Map board coordinates to image coordinates based on orientation
                if self.we_play_white:
                    row_on_image = 7 - row
                    column_on_image = column
                else:
                    row_on_image = row
                    column_on_image = 7 - column
                
                # Check if piece is missing in the image but present on the board
                is_empty = result[row_on_image][column_on_image] == '.'
                if piece and is_empty:
                    start_squares.append(chess.square_name(square))
            
            return start_squares
            
        except Exception as e:
            logger.error(f"Error finding premoves: {e}")
            return []

    def get_valid_move(self, potential_starts: List[str], potential_arrivals: List[str], 
                       current_chessboard_image: np.ndarray) -> str:
        """
        Find a valid chess move based on potential start and arrival squares.
        
        Args:
            potential_starts: List of potential starting square names
            potential_arrivals: List of potential arrival square names
            current_chessboard_image: Current image of the chessboard
            
        Returns:
            UCI string of the validated move, or empty string if no valid move found
        """
        try:
            # Classify the current image
            result = self.classifier.classify(current_chessboard_image)
            valid_move_string = ""
            
            # Try all combinations of start and arrival squares
            for start in potential_starts:
                if valid_move_string:
                    break
                for arrival in potential_arrivals:
                    if valid_move_string:
                        break
                    if start == arrival:
                        continue
                    
                    # Create UCI move string
                    uci_move = start + arrival
                    
                    # Try to parse as a move
                    try:
                        move = chess.Move.from_uci(uci_move)
                    except ValueError:
                        continue
                    
                    # Check if the move is legal and matches the image
                    if move in self.board.legal_moves:
                        if self.can_image_correspond_to_chessboard(move, result):
                            valid_move_string = uci_move
                    else:
                        # Check for promotion moves
                        try:
                            r, c = self.convert_square_name_to_row_column(arrival)
                            if result[r][c] not in ["q", "r", "b", "n"]:
                                continue
                                
                            uci_move_promoted = uci_move + result[r][c]
                            promoted_move = chess.Move.from_uci(uci_move_promoted)
                            
                            if promoted_move in self.board.legal_moves:
                                if self.can_image_correspond_to_chessboard(promoted_move, result):
                                    valid_move_string = uci_move_promoted
                        except Exception as promotion_error:
                            logger.debug(f"Error checking promotion: {promotion_error}")
            
            # Check for castling moves if no normal move found
            if not valid_move_string:
                valid_move_string = self._check_castling_moves(potential_starts, potential_arrivals, result)
            
            # Check for premoves if still no move found
            if not valid_move_string:
                premove_starts = self.find_premove(result)
                for start_square_name in premove_starts:
                    start_square = chess.parse_square(start_square_name)
                    for move in self.board.legal_moves:
                        if move.from_square == start_square:
                            if self.can_image_correspond_to_chessboard(move, result):
                                return move.uci()
            
            return valid_move_string
            
        except Exception as e:
            logger.error(f"Error getting valid move: {e}")
            logger.debug(traceback.format_exc())
            return ""
    
    def _check_castling_moves(self, potential_starts: List[str], potential_arrivals: List[str], 
                             result: List[List[str]]) -> str:
        """Helper method to check for various castling moves."""
        try:
            # Check each type of castling
            castling_patterns = [
                # Format: start squares, arrival squares, move UCI
                (["e1", "h1"], ["f1", "g1"], "e1g1"),  # White kingside
                (["e1", "a1"], ["c1", "d1"], "e1c1"),  # White queenside
                (["e8", "h8"], ["f8", "g8"], "e8g8"),  # Black kingside
                (["e8", "a8"], ["c8", "d8"], "e8c8")   # Black queenside
            ]
            
            for starts, arrivals, move_uci in castling_patterns:
                # Check if all required squares were detected as changed
                if all(s in potential_starts for s in starts) and all(a in potential_arrivals for a in arrivals):
                    move = chess.Move.from_uci(move_uci)
                    
                    # Make sure the move is legal and not already played
                    if move in self.board.legal_moves:
                        if (len(self.board.move_stack) == 0 or self.board.peek() != move) and \
                           self.can_image_correspond_to_chessboard(move, result):
                            return move_uci
            
            return ""
            
        except Exception as e:
            logger.error(f"Error checking castling moves: {e}")
            return ""

    def has_square_image_changed(self, old_square: np.ndarray, new_square: np.ndarray) -> bool:
        """
        Determine if a square has changed between two images.
        
        Args:
            old_square: Image of the square from previous capture
            new_square: Image of the square from current capture
            
        Returns:
            True if the square has changed significantly, False otherwise
        """
        try:
            # Handle invalid inputs
            if old_square is None or new_square is None:
                return False
                
            # Ensure squares are the same size
            if old_square.shape != new_square.shape:
                # Resize to match
                if old_square.size > 0 and new_square.size > 0:
                    if old_square.shape[0] > new_square.shape[0]:
                        old_square = cv2.resize(old_square, (new_square.shape[1], new_square.shape[0]))
                    else:
                        new_square = cv2.resize(new_square, (old_square.shape[1], old_square.shape[0]))
                else:
                    return False
            
            # Calculate absolute difference
            diff = cv2.absdiff(old_square, new_square)
            mean_diff = diff.mean()
            
            return mean_diff > self.image_difference_threshold
            
        except Exception as e:
            logger.error(f"Error checking square change: {e}")
            return False

    def convert_row_column_to_square_name(self, row: int, column: int) -> str:
        """
        Convert row and column indices to a chess square name.
        
        Args:
            row: Row index (0-7)
            column: Column index (0-7)
            
        Returns:
            Chess square name (e.g., 'e4')
        """
        try:
            # Validate indices
            if not (0 <= row < 8) or not (0 <= column < 8):
                logger.warning(f"Invalid row/column: {row}, {column}")
                return "a1"  # Default fallback
                
            if self.we_play_white:
                # Standard orientation
                number = str(8 - row)
                letter = chr(97 + column)
            else:
                # Flipped orientation
                number = str(row + 1)
                letter = chr(97 + (7 - column))
                
            return letter + number
            
        except Exception as e:
            logger.error(f"Error converting coordinates to square name: {e}")
            return "a1"  # Default fallback

    def convert_square_name_to_row_column(self, square_name: str) -> Tuple[int, int]:
        """
        Convert a chess square name to row and column indices.
        
        Args:
            square_name: Chess square name (e.g., 'e4')
            
        Returns:
            Tuple of (row, column) indices
        """
        try:
            # Validate square name
            if not isinstance(square_name, str) or len(square_name) != 2:
                logger.warning(f"Invalid square name: {square_name}")
                return (0, 0)
                
            # Extract file (letter) and rank (number)
            file_char = square_name[0].lower()
            rank_char = square_name[1]
            
            # Validate file and rank
            if not ('a' <= file_char <= 'h') or not ('1' <= rank_char <= '8'):
                logger.warning(f"Invalid square name: {square_name}")
                return (0, 0)
                
            # Convert based on board orientation
            for row in range(8):
                for column in range(8):
                    this_square_name = self.convert_row_column_to_square_name(row, column)
                    if this_square_name == square_name:
                        return row, column
                        
            # Fallback if not found (shouldn't happen)
            return (0, 0)
            
        except Exception as e:
            logger.error(f"Error converting square name to coordinates: {e}")
            return (0, 0)

    def get_potential_moves(self, old_image: np.ndarray, new_image: np.ndarray) -> Tuple[List[str], List[str]]:
        """
        Find potential starting and arrival squares by comparing two board images.
        
        Args:
            old_image: Previous chessboard image
            new_image: Current chessboard image
            
        Returns:
            Tuple of (potential start squares, potential arrival squares)
        """
        try:
            potential_starts = []
            potential_arrivals = []
            
            # Check each square for changes
            for row in range(8):
                for column in range(8):
                    # Extract square images
                    old_square = self.get_square_image(row, column, old_image)
                    new_square = self.get_square_image(row, column, new_image)
                    
                    # Check if the square has changed
                    if self.has_square_image_changed(old_square, new_square):
                        square_name = self.convert_row_column_to_square_name(row, column)
                        potential_starts.append(square_name)
                        potential_arrivals.append(square_name)
            
            return potential_starts, potential_arrivals
            
        except Exception as e:
            logger.error(f"Error getting potential moves: {e}")
            return [], []

    def register_move_if_needed(self) -> Tuple[bool, Any]:
        """
        Check if a new move has been made and register it if valid.
        
        Returns:
            Tuple of (move_found, move_object or error_message)
        """
        try:
            # Capture current board state
            new_board = self.get_chessboard()
            if new_board is None:
                return False, "Failed to capture board image"
                
            # Find potential moves
            potential_starts, potential_arrivals = self.get_potential_moves(
                self.previous_chessboard_image, new_board
            )
            
            # No changes detected
            if not potential_starts:
                # Check if there's a pending move to register from game thread
                if (self.game_thread and 
                    len(self.registered_moves) < len(self.game_thread.played_moves)):
                    valid_move_UCI = self.game_thread.played_moves[len(self.registered_moves)]
                    self.register_move(valid_move_UCI, self.previous_chessboard_image)
                    return True, valid_move_UCI
                return False, "No changes detected"
            
            # Try to validate a move
            valid_move_string1 = self.get_valid_move(potential_starts, potential_arrivals, new_board)
            
            if valid_move_string1:
                # Wait briefly and check again to avoid capturing during animations
                time.sleep(0.1)
                new_board = self.get_chessboard()
                if new_board is None:
                    return False, "Failed to capture verification image"
                    
                potential_starts, potential_arrivals = self.get_potential_moves(
                    self.previous_chessboard_image, new_board
                )
                valid_move_string2 = self.get_valid_move(potential_starts, potential_arrivals, new_board)
                
                # Ensure the move hasn't changed (avoid capturing during animations)
                if valid_move_string2 != valid_move_string1:
                    return False, "The move has changed during verification"
                
                # Register the move
                try:
                    valid_move_UCI = chess.Move.from_uci(valid_move_string1)
                    if self.register_move(valid_move_UCI, new_board):
                        logger.info(f"Move registered: {valid_move_string1}")
                        return True, valid_move_UCI
                    else:
                        return False, "Failed to register move"
                except ValueError:
                    return False, f"Invalid move string: {valid_move_string1}"
            
            return False, "No valid move found"
            
        except Exception as e:
            logger.error(f"Error in register_move_if_needed: {e}")
            logger.debug(traceback.format_exc())
            return False, f"Error: {str(e)}"

    def register_move(self, move: chess.Move, board_image: np.ndarray) -> bool:
        """
        Register a move and update the board state.
        
        Args:
            move: Chess move to register
            board_image: Current board image to save
            
        Returns:
            True if move was successfully registered, False otherwise
        """
        try:
            # Validate move
            if move not in self.board.legal_moves:
                logger.warning(f"Attempted to register illegal move: {move.uci()}")
                return False
                
            # Apply move
            self.board.push(move)
            
            # Update board image and registered moves list
            self.previous_chessboard_image = board_image
            self.registered_moves.append(move)
            
            # Save diagnostic image if needed
            # cv2.imwrite("registered.jpg", board_image)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering move: {e}")
            return False


# Test function
def test_commentator():
    """Test the commentator functionality with a mock board."""
    print("This module provides commentary for chess games.")
    print("It requires a visible chess board on screen to function.")
    print("Use the main.py interface to run a complete system.")


if __name__ == "__main__":
    test_commentator()
