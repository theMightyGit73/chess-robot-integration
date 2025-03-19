import chessboard_detection
import pyautogui
import time
import chess
import logging
import traceback
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InternetGame")

class Internet_game:
    """
    Manages interaction with online chess games by detecting the board position on screen
    and simulating mouse movements to make moves.
    """
    
    def __init__(self, use_template: bool = True, start_delay: int = 5, drag_drop: bool = False):
        """
        Initialize the Internet Game interface.
        
        Args:
            use_template: Whether to use template matching (True) or auto-detection (False)
            start_delay: Delay in seconds before starting board detection
            drag_drop: Whether to use drag and drop (True) or click-click (False) for moves
            
        Raises:
            ValueError: If board detection fails
        """
        self.drag_drop = drag_drop
        self.position = None
        self.we_play_white = False
        self.is_our_turn = False
        self.detection_attempts = 3  # Number of detection attempts
        self.click_duration = 0.1     # Duration for mouse clicks
        self.drag_duration = 0.3      # Duration for drag movements
        
        # Record start time for logging
        start_time = time.time()
        logger.info(f"Initializing Internet Game with {start_delay}s delay")
        
        try:
            # Apply initial delay to allow switching to the game window
            logger.info(f"Waiting {start_delay} seconds before detecting board...")
            time.sleep(start_delay)
            
            # Try board detection with multiple attempts if needed
            for attempt in range(self.detection_attempts):
                try:
                    if use_template:
                        logger.info("Using template matching for board detection")
                        self.position, self.we_play_white = chessboard_detection.find_chessboard()
                    else:
                        logger.info("Using automatic board detection")
                        self.position, self.we_play_white = chessboard_detection.auto_find_chessboard()
                    
                    # Check if detection was successful
                    if self.position:
                        self.is_our_turn = self.we_play_white
                        logger.info(f"Board detected successfully. Playing as {'white' if self.we_play_white else 'black'}")
                        logger.info(f"Board position: ({self.position.minX}, {self.position.minY}) to ({self.position.maxX}, {self.position.maxY})")
                        break
                    else:
                        logger.warning(f"Board detection attempt {attempt+1} failed, retrying...")
                        time.sleep(1)  # Short delay before retry
                        
                except Exception as e:
                    logger.error(f"Detection error on attempt {attempt+1}: {e}")
                    if attempt < self.detection_attempts - 1:
                        time.sleep(1)  # Short delay before retry
            
            # Check if board was successfully detected
            if not self.position:
                error_msg = "Failed to detect chess board after multiple attempts"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Log total initialization time
            elapsed = time.time() - start_time - start_delay
            logger.info(f"Board detection completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing game: {e}")
            traceback.print_exc()
            raise ValueError(f"Game initialization failed: {e}")

    def move(self, move: chess.Move) -> bool:
        """
        Execute a move on the online chess board by simulating mouse movements.
        
        Args:
            move: Chess move to execute
            
        Returns:
            True if the move was successfully executed, False otherwise
            
        Raises:
            ValueError: If move is invalid or coordinates cannot be determined
        """
        if not move:
            logger.error("Invalid move: None")
            return False
            
        move_string = move.uci()
        logger.info(f"Executing move: {move_string}")
        
        try:
            # Extract origin and destination squares
            if len(move_string) < 4:
                logger.error(f"Invalid move string: {move_string}")
                return False
                
            origin_square = move_string[0:2]
            destination_square = move_string[2:4]
            
            # Get coordinates for squares
            try:
                center_x_origin, center_y_origin = self.get_square_center(origin_square)
                center_x_dest, center_y_dest = self.get_square_center(destination_square)
            except Exception as e:
                logger.error(f"Error getting square coordinates: {e}")
                return False
            
            # Log planned move coordinates
            logger.debug(f"Move coordinates: ({center_x_origin}, {center_y_origin}) to ({center_x_dest}, {center_y_dest})")
            
            # Execute move with appropriate method
            success = False
            for attempt in range(2):  # Try up to twice
                try:
                    if self.drag_drop:
                        # Execute move as drag and drop
                        pyautogui.moveTo(center_x_origin, center_y_origin, duration=0.01)
                        pyautogui.mouseDown(center_x_origin, center_y_origin, button='left')
                        pyautogui.moveTo(center_x_dest, center_y_dest, duration=self.drag_duration)
                        pyautogui.mouseUp(button='left')
                    else:
                        # Execute move as click-click
                        pyautogui.click(center_x_origin, center_y_origin, duration=self.click_duration)
                        pyautogui.click(center_x_dest, center_y_dest, duration=self.click_duration)
                    
                    success = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Move execution attempt {attempt+1} failed: {e}")
                    time.sleep(0.5)  # Short delay before retry
            
            if success:
                logger.info(f"Move {origin_square} to {destination_square} executed successfully")
                return True
            else:
                logger.error(f"Failed to execute move after multiple attempts")
                return False
                
        except Exception as e:
            logger.error(f"Error executing move: {e}")
            traceback.print_exc()
            return False

    def get_square_center(self, square_name: str) -> Tuple[int, int]:
        """
        Calculate the screen coordinates for the center of a chess square.
        
        Args:
            square_name: Chess square name (e.g., 'e4')
            
        Returns:
            Tuple of (x, y) coordinates for the center of the square
            
        Raises:
            ValueError: If square name is invalid or position is not detected
        """
        if not self.position:
            raise ValueError("Board position not detected")
            
        try:
            # Validate square name
            if not isinstance(square_name, str) or len(square_name) != 2:
                raise ValueError(f"Invalid square name: {square_name}")
                
            # Convert square name to row and column
            row, column = self.convert_square_name_to_row_column(square_name, self.we_play_white)
            
            # Calculate center coordinates
            center_x = int(self.position.minX + (column + 0.5) * (self.position.maxX - self.position.minX) / 8)
            center_y = int(self.position.minY + (row + 0.5) * (self.position.maxY - self.position.minY) / 8)
            
            return center_x, center_y
            
        except Exception as e:
            logger.error(f"Error calculating square center for {square_name}: {e}")
            raise ValueError(f"Cannot determine center for square {square_name}: {e}")

    def convert_square_name_to_row_column(self, square_name: str, is_white_on_bottom: bool) -> Tuple[int, int]:
        """
        Convert a chess square name to row and column indices based on board orientation.
        
        Args:
            square_name: Chess square name (e.g., 'e4')
            is_white_on_bottom: Whether white pieces are on the bottom of the screen
            
        Returns:
            Tuple of (row, column) indices (0-7)
            
        Raises:
            ValueError: If square name is invalid
        """
        try:
            # Validate square name format
            if not isinstance(square_name, str) or len(square_name) != 2:
                raise ValueError(f"Invalid square name: {square_name}")
                
            # Extract file (letter) and rank (number)
            file_char = square_name[0].lower()
            rank_char = square_name[1]
            
            # Validate file and rank
            if not ('a' <= file_char <= 'h') or not ('1' <= rank_char <= '8'):
                raise ValueError(f"Invalid square name: {square_name}")
                
            file_idx = ord(file_char) - ord('a')  # 0-7 for a-h
            rank_idx = int(rank_char) - 1  # 0-7 for 1-8
            
            if is_white_on_bottom:
                # Standard orientation (a1 is bottom-left)
                row = 7 - rank_idx
                column = file_idx
            else:
                # Flipped orientation (a1 is top-right)
                row = rank_idx
                column = 7 - file_idx
                
            return row, column
            
        except Exception as e:
            logger.error(f"Error converting square name {square_name}: {e}")
            # Return default as last resort to avoid crashes
            return 0, 0

    def convert_row_column_to_square_name(self, row: int, column: int, is_white_on_bottom: bool) -> str:
        """
        Convert row and column indices to a chess square name based on board orientation.
        
        Args:
            row: Row index (0-7)
            column: Column index (0-7)
            is_white_on_bottom: Whether white pieces are on the bottom of the screen
            
        Returns:
            Chess square name (e.g., 'e4')
            
        Raises:
            ValueError: If row or column is out of range
        """
        try:
            # Validate row and column
            if not (0 <= row <= 7) or not (0 <= column <= 7):
                raise ValueError(f"Invalid row/column: {row}, {column}")
                
            if is_white_on_bottom:
                # Standard orientation (a1 is bottom-left)
                rank = 8 - row  # 8-1 for rows 0-7
                file = chr(97 + column)  # a-h for columns 0-7
            else:
                # Flipped orientation (a1 is top-right)
                rank = row + 1  # 1-8 for rows 0-7
                file = chr(97 + (7 - column))  # a-h for columns 7-0
                
            return f"{file}{rank}"
            
        except Exception as e:
            logger.error(f"Error converting row/column ({row}, {column}): {e}")
            # Return empty string as fallback
            return ""

    def get_board_dimensions(self) -> Tuple[int, int, int, int]:
        """
        Get the dimensions of the detected chess board.
        
        Returns:
            Tuple of (minX, minY, maxX, maxY)
            
        Raises:
            ValueError: If board position is not detected
        """
        if not self.position:
            raise ValueError("Board position not detected")
            
        return (
            self.position.minX,
            self.position.minY,
            self.position.maxX,
            self.position.maxY
        )
        
    def set_click_duration(self, duration: float) -> None:
        """Set the duration for mouse clicks."""
        if duration > 0:
            self.click_duration = duration
            logger.info(f"Click duration set to {duration}s")
            
    def set_drag_duration(self, duration: float) -> None:
        """Set the duration for drag movements."""
        if duration > 0:
            self.drag_duration = duration
            logger.info(f"Drag duration set to {duration}s")


# Test function
def test_internet_game():
    """Test the Internet_game class functionality."""
    try:
        # Initialize with longer delay to allow positioning
        print("Please switch to a chess website within 10 seconds...")
        game = Internet_game(use_template=False, start_delay=10, drag_drop=False)
        
        # Print detected board info
        print(f"Board detected at: {game.get_board_dimensions()}")
        print(f"Playing as {'white' if game.we_play_white else 'black'}")
        
        # Test square coordinate calculation
        for square in ['a1', 'h1', 'a8', 'h8', 'e4']:
            try:
                x, y = game.get_square_center(square)
                print(f"Square {square} center: ({x}, {y})")
            except ValueError as e:
                print(f"Error getting center for {square}: {e}")
        
        # Ask if user wants to test a move
        test_move = input("Test a move? (y/n): ")
        if test_move.lower() == 'y':
            from_square = input("From square (e.g., e2): ")
            to_square = input("To square (e.g., e4): ")
            
            try:
                move = chess.Move.from_uci(f"{from_square}{to_square}")
                game.move(move)
                print("Move executed")
            except ValueError as e:
                print(f"Error executing move: {e}")
        
        print("Test completed successfully")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_internet_game()
