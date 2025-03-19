import berserk
import sys
import os
import chess
import pickle
import time
import logging
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LichessGame")

class Lichess_game:
    """
    Handles interaction with the Lichess API for playing chess games.
    Manages game state, move execution, and reconnection logic.
    """
    
    def __init__(self, token: str):
        """
        Initialize a connection to an ongoing Lichess game.
        
        Args:
            token: Lichess API access token
            
        Raises:
            ValueError: If token is invalid or no/multiple games are found
            ConnectionError: If connection to Lichess fails
        """
        if not token or not isinstance(token, str):
            raise ValueError("Valid Lichess API token is required")
            
        self.token = token
        self.client = None
        self.game_id = None
        self.we_play_white = False
        self.is_our_turn = False
        self.save_file = os.path.join(os.path.expanduser("~"), ".chess_robot", "promotion.bin")
        
        # Map promotion piece names to chess library constants
        self.promotion_pieces: Dict[str, int] = {
            "Queen": chess.QUEEN,
            "Knight": chess.KNIGHT,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP
        }
        
        # Default promotion piece
        self.default_promotion = chess.QUEEN
        
        # Connect to Lichess and find the current game
        self._connect_to_lichess()
        self._find_current_game()
        
        # Create directory for save file if it doesn't exist
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        
        logger.info(f"Connected to Lichess game: {self.game_id}")
        logger.info(f"Playing as {'white' if self.we_play_white else 'black'}")
    
    def _connect_to_lichess(self, max_retries: int = 3) -> None:
        """
        Establish connection to Lichess API with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            
        Raises:
            ConnectionError: If connection cannot be established after retries
        """
        for attempt in range(max_retries):
            try:
                # Try different ways to initialize TokenSession based on berserk version
                try:
                    # For newer versions of berserk that don't accept encoding parameter
                    session = berserk.TokenSession(self.token)
                except TypeError:
                    try:
                        # For older versions of berserk that might require encoding
                        session = berserk.TokenSession(token=self.token)
                    except:
                        # Last resort attempt with minimal arguments
                        session = berserk.session.TokenSession(self.token)
                
                self.client = berserk.Client(session)
                
                # Test connection with a simple API call
                self.client.account.get()
                logger.debug("Successfully connected to Lichess API")
                return
                
            except berserk.exceptions.ResponseError as e:
                if "Invalid token" in str(e) or getattr(e, 'status_code', 0) == 401:
                    logger.error(f"Invalid token provided: {e}")
                    raise ValueError(f"Invalid Lichess API token: {e}")
                
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
            
            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        
        raise ConnectionError("Failed to connect to Lichess API after multiple attempts")
    
    def _find_current_game(self) -> None:
        """
        Find the current ongoing game on Lichess.
        
        Raises:
            ValueError: If no games or multiple games are found
        """
        try:
            games = self.client.games.get_ongoing()
            
            if not games:
                logger.error("No ongoing games found on Lichess")
                raise ValueError("No games found. Please create your game on Lichess.")
                
            if len(games) > 1:
                logger.error(f"Multiple ongoing games found: {len(games)}")
                raise ValueError("Multiple games found. Please make sure there is only one ongoing game on Lichess.")
            
            game = games[0]
            self.game_id = game['gameId']
            self.we_play_white = game['color'] == 'white'
            self.is_our_turn = self.we_play_white
            
            # Get additional game info for logging
            try:
                game_info = self.client.games.get_ongoing_by_id(self.game_id)
                opponent = game_info.get('opponent', {}).get('username', 'Unknown')
                logger.info(f"Playing against: {opponent}")
                
                # Log time control if available
                if 'clock' in game_info:
                    clock = game_info['clock']
                    time_control = f"{clock.get('initial')/60}+{clock.get('increment')}"
                    logger.info(f"Time control: {time_control}")
            except Exception as e:
                logger.warning(f"Could not fetch detailed game info: {e}")
            
        except Exception as e:
            if not isinstance(e, ValueError):
                logger.error(f"Error finding current game: {e}")
                raise ValueError(f"Could not find current game: {e}")
            raise
    
    def _load_promotion_choice(self) -> int:
        """
        Load user's promotion piece preference from file.
        
        Returns:
            Chess piece constant for promotion
        """
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'rb') as infile:
                    piece_name = pickle.load(infile)
                    if piece_name in self.promotion_pieces:
                        return self.promotion_pieces[piece_name]
                    else:
                        logger.warning(f"Unknown promotion piece: {piece_name}, using Queen")
        except Exception as e:
            logger.warning(f"Could not load promotion choice: {e}")
        
        return self.default_promotion
    
    def move(self, move: chess.Move) -> bool:
        """
        Make a move on the Lichess board.
        
        Args:
            move: Chess move to execute
            
        Returns:
            True if the move was successful, False otherwise
        """
        # Handle promotion if needed
        if move.promotion:
            move.promotion = self._load_promotion_choice()
        
        move_string = move.uci()
        logger.info(f"Attempting to play move: {move_string}")
        
        # Try to make the move with retry logic
        for attempt in range(3):
            try:
                self.client.board.make_move(self.game_id, move_string)
                logger.info(f"Successfully played move: {move_string}")
                return True
                
            except berserk.exceptions.ResponseError as e:
                if "not your turn" in str(e).lower():
                    logger.warning("Not your turn to move")
                    self.is_our_turn = False
                    return False
                    
                elif "game over" in str(e).lower():
                    logger.info("Game is already over")
                    return False
                
                elif "invalid move" in str(e).lower():
                    logger.error(f"Invalid move: {move_string}")
                    return False
                
                else:
                    logger.warning(f"API error on attempt {attempt+1}/3: {e}")
            
            except Exception as e:
                logger.warning(f"Error on attempt {attempt+1}/3: {e}")
            
            # Attempt to reconnect before retrying
            try:
                logger.info("Attempting to reconnect to Lichess...")
                session = berserk.TokenSession(self.token)
                self.client = berserk.Client(session)
                logger.info("Reconnected to Lichess")
            except Exception as reconnect_error:
                logger.warning(f"Reconnection failed: {reconnect_error}")
            
            # Wait before retry
            time.sleep(1)
        
        logger.error(f"Failed to make move {move_string} after multiple attempts")
        return False
    
    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the current state of the game from Lichess.
        
        Returns:
            Dictionary with game state or None if request fails
        """
        try:
            return self.client.board.get_game(self.game_id)
        except Exception as e:
            logger.error(f"Failed to get game state: {e}")
            return None
    
    def resign(self) -> bool:
        """
        Resign the current game.
        
        Returns:
            True if resignation was successful, False otherwise
        """
        try:
            self.client.board.resign_game(self.game_id)
            logger.info("Resigned from game")
            return True
        except Exception as e:
            logger.error(f"Failed to resign: {e}")
            return False
    
    def offer_draw(self) -> bool:
        """
        Offer a draw in the current game.
        
        Returns:
            True if draw offer was sent, False otherwise
        """
        try:
            self.client.board.offer_draw(self.game_id)
            logger.info("Draw offered")
            return True
        except Exception as e:
            logger.error(f"Failed to offer draw: {e}")
            return False


# Test function
def test_lichess_connection(token):
    """Test connection to Lichess and basic functionality."""
    try:
        print("Testing Lichess connection...")
        game = Lichess_game(token)
        print(f"Successfully connected to game: {game.game_id}")
        print(f"Playing as {'white' if game.we_play_white else 'black'}")
        
        # Get current game state
        state = game.get_game_state()
        if state:
            print(f"Game status: {state.get('status', 'unknown')}")
            
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


# Run test if script is executed directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_token = sys.argv[1]
        test_lichess_connection(test_token)
    else:
        print("Please provide a Lichess API token as an argument")
        print("Usage: python lichess_game.py YOUR_TOKEN")
