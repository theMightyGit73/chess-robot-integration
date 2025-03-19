import chess
import logging
import time
import traceback
from threading import Thread
from typing import Optional, Tuple, Iterator, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LichessCommentator")

class Lichess_commentator(Thread):
    """
    Thread that monitors a Lichess game stream and provides spoken commentary
    on moves as they occur in the game.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Lichess commentator thread."""
        super(Lichess_commentator, self).__init__(*args, **kwargs)
        self.stream: Optional[Iterator] = None
        self.speech_thread = None
        self.game_state = Game_state()
        self.comment_me: bool = False
        self.comment_opponent: bool = False
        self.language = None
        self.running: bool = True
        self.error_count: int = 0
        self.max_errors: int = 10  # Maximum consecutive errors before thread exits
        self.last_move_time: float = time.time()
        
    def run(self):
        """
        Main thread loop that monitors the game stream and triggers commentary.
        """
        if self.stream is None:
            logger.error("No game stream provided")
            return
            
        logger.info("Lichess commentator started")
        
        while self.running and not self.game_state.board.is_game_over() and not self.game_state.resign_or_draw:
            try:
                # Determine whose turn it is
                is_my_turn = (self.game_state.we_play_white) == (self.game_state.board.turn == chess.WHITE)
                
                # Check for new moves
                found_move, move = self.game_state.register_move_if_needed(self.stream)
                
                if found_move:
                    self.last_move_time = time.time()
                    self.error_count = 0  # Reset error counter on successful move
                    
                    # Determine if we should comment on this move
                    should_comment = (
                        (self.comment_me and is_my_turn) or 
                        (self.comment_opponent and not is_my_turn)
                    )
                    
                    if should_comment and self.speech_thread:
                        try:
                            comment = self.language.comment(self.game_state.board, move)
                            logger.debug(f"Speaking comment: {comment}")
                            self.speech_thread.put_text(comment)
                        except Exception as comment_error:
                            logger.error(f"Error generating or speaking comment: {comment_error}")
                
                # Brief pause to prevent high CPU usage
                time.sleep(0.1)
                
                # Check for game timeout
                if time.time() - self.last_move_time > 300:  # 5 minutes
                    logger.warning("No moves for 5 minutes, checking game status")
                    self.check_game_status()
                    self.last_move_time = time.time()  # Reset timer after check
                
            except StopIteration:
                logger.info("Game stream ended")
                break
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in commentator loop: {e}")
                
                if self.error_count >= self.max_errors:
                    logger.critical(f"Too many consecutive errors ({self.max_errors}), stopping commentator")
                    break
                    
                # Progressively longer pauses on repeated errors
                time.sleep(min(5, 0.5 * self.error_count))
        
        # Game ended
        self.handle_game_end()
        logger.info("Lichess commentator stopped")
    
    def check_game_status(self):
        """Check if the game is still active via API call."""
        try:
            if hasattr(self.game_state.game, 'internet_game') and hasattr(self.game_state.game.internet_game, 'client'):
                client = self.game_state.game.internet_game.client
                game_id = self.game_state.game.internet_game.game_id
                
                game_info = client.board.get_game(game_id)
                if game_info.get('status') != 'started':
                    logger.info(f"Game is no longer active: {game_info.get('status')}")
                    self.game_state.resign_or_draw = True
        except Exception as e:
            logger.warning(f"Failed to check game status: {e}")
    
    def handle_game_end(self):
        """Handle the end of the game with appropriate commentary."""
        try:
            if not self.game_state.board.is_game_over() and self.game_state.resign_or_draw:
                if self.speech_thread and self.language:
                    self.speech_thread.put_text(self.language.game_over_resignation_or_draw)
            elif self.game_state.board.is_game_over():
                result = self.game_state.board.result()
                
                if self.speech_thread and self.language:
                    if result == "1-0":  # White wins
                        self.speech_thread.put_text(self.language.white_wins)
                    elif result == "0-1":  # Black wins
                        self.speech_thread.put_text(self.language.black_wins)
                    elif result == "1/2-1/2":  # Draw
                        self.speech_thread.put_text(self.language.draw)
                    else:
                        self.speech_thread.put_text(self.language.game_over)
                
                logger.info(f"Game ended: {result}")
        except Exception as e:
            logger.error(f"Error handling game end: {e}")
    
    def stop(self):
        """Gracefully stop the commentator thread."""
        logger.info("Stopping commentator thread")
        self.running = False


class Game_state:
    """
    Maintains the state of a chess game and processes incoming moves
    from the Lichess game stream.
    """

    def __init__(self):
        """Initialize the game state."""
        self.we_play_white: Optional[bool] = None
        self.board = chess.Board()
        self.registered_moves: List[chess.Move] = []
        self.resign_or_draw: bool = False
        self.game = None
        self.variant: str = 'wait'
        self.last_state_update: float = time.time()
    
    def register_move_if_needed(self, stream: Iterator) -> Tuple[bool, Any]:
        """
        Check the game stream for new moves and register them.
        
        Args:
            stream: Iterator for the Lichess game stream
            
        Returns:
            Tuple of (move_found, move_object)
        """
        try:
            current_state = next(stream)
            self.last_state_update = time.time()
            
            # Handle initial state setup
            if 'state' in current_state:
                self._handle_initial_state(current_state)
                current_state = current_state['state']
            
            # Process moves
            if 'moves' in current_state:
                # Process new moves
                result = self._process_moves(current_state['moves'])
                if result[0]:  # If a new move was found
                    return result
                
            # Check game status
            if 'status' in current_state and current_state['status'] in ["resign", "draw", "aborted", "mate", "timeout"]:
                logger.info(f"Game ending with status: {current_state['status']}")
                self.resign_or_draw = True
            
        except StopIteration:
            raise  # Re-raise to signal stream end
            
        except Exception as e:
            logger.error(f"Error processing game state: {e}")
            logger.debug(traceback.format_exc())
        
        return False, "No move found"
    
    def _handle_initial_state(self, current_state: Dict[str, Any]) -> None:
        """
        Handle the initial game state and set up the board.
        
        Args:
            current_state: Dictionary with the initial game state
        """
        try:
            # Set variant based on initial position
            if 'initialFen' in current_state:
                initial_fen = current_state['initialFen']
                
                if initial_fen == 'startpos':
                    self.variant = 'standard'
                    logger.info("Standard chess variant detected")
                else:
                    self.variant = 'fromPosition'
                    logger.info(f"Custom position detected: {initial_fen}")
                    self.from_position(initial_fen)
        except Exception as e:
            logger.error(f"Error handling initial state: {e}")
    
    def _process_moves(self, moves_str: str) -> Tuple[bool, Any]:
        """
        Process moves from the game state.
        
        Args:
            moves_str: Space-separated string of moves in UCI format
            
        Returns:
            Tuple of (move_found, move_object)
        """
        try:
            moves = moves_str.split()
            
            # Check for new moves
            if len(moves) > len(self.registered_moves):
                valid_move_string = moves[len(self.registered_moves)]
                
                try:
                    valid_move_UCI = chess.Move.from_uci(valid_move_string)
                    
                    # Register the new move
                    if self.register_move(valid_move_UCI):
                        logger.info(f"New move registered: {valid_move_string}")
                        return True, valid_move_UCI
                    else:
                        logger.warning(f"Invalid move: {valid_move_string}")
                except ValueError as e:
                    logger.error(f"Error parsing UCI move: {valid_move_string} - {e}")
            
            # Check for takeback or move adjustments
            while len(moves) < len(self.registered_moves):
                logger.info("Move takeback detected")
                self.unregister_move()
                
        except Exception as e:
            logger.error(f"Error processing moves: {e}")
        
        return False, "No move found"

    def register_move(self, move: chess.Move) -> bool:
        """
        Register a move on the board.
        
        Args:
            move: Chess move object
            
        Returns:
            True if the move was successfully registered, False otherwise
        """
        try:
            if move in self.board.legal_moves:
                self.board.push(move)
                self.registered_moves.append(move)
                return True
            else:
                logger.warning(f"Attempted to register illegal move: {move.uci()}")
                return False
        except Exception as e:
            logger.error(f"Error registering move: {e}")
            return False

    def unregister_move(self) -> None:
        """Unregister the last move (used for takebacks or corrections)."""
        try:
            if len(self.registered_moves) > 0:
                self.board.pop()
                removed_move = self.registered_moves.pop()
                logger.info(f"Unregistered move: {removed_move.uci()}")
                
                # Synchronize with game state
                if self.game and len(self.registered_moves) < len(self.game.executed_moves):
                    self.game.executed_moves.pop()
                    self.game.played_moves.pop()
                    self.game.board.pop()
                    
                    if hasattr(self.game, 'internet_game'):
                        self.game.internet_game.is_our_turn = not self.game.internet_game.is_our_turn
            else:
                logger.warning("Attempted to unregister move but no moves are registered")
                
        except Exception as e:
            logger.error(f"Error unregistering move: {e}")

    def from_position(self, fen: str) -> None:
        """
        Set up the board from a custom position.
        
        Args:
            fen: FEN string representing the board position
        """
        try:
            # Validate FEN string
            if not self._is_valid_fen(fen):
                logger.warning(f"Invalid FEN string: {fen}")
                return
                
            # Set up board
            self.board = chess.Board(fen)
            
            # Synchronize with game state
            if self.game:
                self.game.board = chess.Board(fen)
                
                # Adjust turn if needed
                if hasattr(self.game, 'internet_game') and self.board.turn == chess.BLACK:
                    self.game.internet_game.is_our_turn = not self.game.internet_game.is_our_turn
                    
            logger.info(f"Board set up from position: {fen}")
            
        except Exception as e:
            logger.error(f"Error setting up board from position: {e}")
    
    def _is_valid_fen(self, fen: str) -> bool:
        """
        Validate a FEN string.
        
        Args:
            fen: FEN string to validate
            
        Returns:
            True if the FEN string is valid, False otherwise
        """
        try:
            chess.Board(fen)
            return True
        except ValueError:
            return False
        except Exception:
            return False


# Test function
def test_commentator():
    """
    Test the Lichess commentator functionality.
    This requires a mock speech thread and game stream.
    """
    from queue import Queue
    
    class MockSpeechThread:
        def __init__(self):
            self.queue = Queue()
        
        def put_text(self, text):
            print(f"Speech: {text}")
            self.queue.put(text)
    
    class MockLanguage:
        def __init__(self):
            self.game_over = "Game over"
            self.white_wins = "White wins"
            self.black_wins = "Black wins"
            self.draw = "It's a draw"
            self.game_over_resignation_or_draw = "Game over by resignation or draw"
            
        def comment(self, board, move):
            return f"Move: {move.uci()}"
    
    # Create mock objects
    speech_thread = MockSpeechThread()
    language = MockLanguage()
    
    # Create commentator
    commentator = Lichess_commentator()
    commentator.speech_thread = speech_thread
    commentator.language = language
    commentator.comment_me = True
    commentator.comment_opponent = True
    
    # Mock game state
    commentator.game_state.we_play_white = True
    
    # Test move registration
    print("Testing move registration...")
    test_move = chess.Move.from_uci("e2e4")
    if commentator.game_state.register_move(test_move):
        print("Move registration successful")
    else:
        print("Move registration failed")
    
    print("Test complete")


if __name__ == "__main__":
    test_commentator()
