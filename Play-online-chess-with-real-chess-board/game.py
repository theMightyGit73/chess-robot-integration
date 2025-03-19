import time
import chess
import cv2
import numpy as np
import pickle
import os
import sys
import traceback
from collections import defaultdict
import gc

from helper import detect_state, get_square_image, predict
from internet_game import Internet_game
from lichess_game import Lichess_game
from commentator import Commentator_thread
from lichess_commentator import Lichess_commentator

# Constants for move detection thresholds
SQUARE_SCORE_THRESHOLD = 10.0
CNN_CONFIDENCE_THRESHOLD = 0.5
HOG_MATCH_THRESHOLD = 0.5
CASTLING_SCORE_BONUS = 10.0

class Game:
    """
    Main game class that handles move detection, validation and execution
    """
    def __init__(self, board_basics, speech_thread, use_template, make_opponent, start_delay, comment_me,
                 comment_opponent, drag_drop, language, token, roi_mask):
        """Initialize the game with all required components"""
        self.board_basics = board_basics
        self.speech_thread = speech_thread
        self.make_opponent = make_opponent
        self.executed_moves = []
        self.played_moves = []
        self.board = chess.Board()
        self.comment_me = comment_me
        self.comment_opponent = comment_opponent
        self.language = language
        self.roi_mask = roi_mask
        
        # ML model configuration
        self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        self.knn = cv2.ml.KNearest_create()
        self.features = None
        self.labels = None
        self.save_file = 'hog.bin'
        
        # Load CNN models
        try:
            self.piece_model = cv2.dnn.readNetFromONNX("cnn_piece.onnx")
            self.color_model = cv2.dnn.readNetFromONNX("cnn_color.onnx")
            print("CNN models loaded successfully")
        except Exception as e:
            print(f"Error loading CNN models: {e}")
            print("Using fallback detection methods")
            self.piece_model = None
            self.color_model = None
        
        # Set up game connection (Lichess API or screen detection)
        if token:
            try:
                self.internet_game = Lichess_game(token)
                print(f"Connected to Lichess game: {self.internet_game.game_id}")
            except Exception as e:
                print(f"Error connecting to Lichess: {e}")
                print("Falling back to screen detection")
                self.internet_game = Internet_game(use_template, start_delay, drag_drop)
        else:
            self.internet_game = Internet_game(use_template, start_delay, drag_drop)
            
        # Initialize appropriate commentator
        self.setup_commentator(token)
        
        # Cache for move detection results to avoid redundant computations
        self.state_cache = {}
        self.move_cache = {}
        
    def setup_commentator(self, token):
        """Set up the appropriate commentator based on connection type"""
        try:
            if token:
                # Lichess API-based commentator
                commentator_thread = Lichess_commentator()
                commentator_thread.daemon = True
                commentator_thread.stream = self.internet_game.client.board.stream_game_state(self.internet_game.game_id)
                commentator_thread.speech_thread = self.speech_thread
                commentator_thread.game_state.we_play_white = self.internet_game.we_play_white
                commentator_thread.game_state.game = self
                commentator_thread.comment_me = self.comment_me
                commentator_thread.comment_opponent = self.comment_opponent
                commentator_thread.language = self.language
            else:
                # Screen detection based commentator
                commentator_thread = Commentator_thread()
                commentator_thread.daemon = True
                commentator_thread.speech_thread = self.speech_thread
                commentator_thread.game_state.game_thread = self
                commentator_thread.game_state.we_play_white = self.internet_game.we_play_white
                commentator_thread.game_state.board_position_on_screen = self.internet_game.position
                commentator_thread.comment_me = self.comment_me
                commentator_thread.comment_opponent = self.comment_opponent
                commentator_thread.language = self.language
            
            self.commentator = commentator_thread
            print("Commentator initialized successfully")
        except Exception as e:
            print(f"Error setting up commentator: {e}")
            self.commentator = None

    def initialize_hog(self, frame):
        """Initialize the HOG detector with the current board state"""
        try:
            print("Initializing HOG detector...")
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pieces = []
            squares = []
            
            # Collect training examples for pieces and empty squares
            for row in range(8):
                for column in range(8):
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    square_image = get_square_image(row, column, frame_gray)
                    square_image = cv2.resize(square_image, (64, 64))
                    if piece:
                        pieces.append(square_image)
                    else:
                        squares.append(square_image)
            
            # Extract HOG features
            pieces_hog = [self.hog.compute(piece) for piece in pieces]
            squares_hog = [self.hog.compute(square) for square in squares]
            
            # Create labels (1 for pieces, 0 for empty squares)
            labels_pieces = np.ones((len(pieces_hog), 1), np.int32)
            labels_squares = np.zeros((len(squares_hog), 1), np.int32)
            
            # Combine features and labels
            pieces_hog = np.array(pieces_hog)
            squares_hog = np.array(squares_hog)
            
            # Handle possible empty arrays
            if len(pieces_hog) == 0 or len(squares_hog) == 0:
                print("Warning: Not enough training examples for HOG detector")
                return False
            
            features = np.float32(np.concatenate((pieces_hog, squares_hog), axis=0))
            labels = np.concatenate((labels_pieces, labels_squares), axis=0)
            
            # Train KNN classifier
            self.knn.train(features, cv2.ml.ROW_SAMPLE, labels)
            self.features = features
            self.labels = labels
            
            # Save the trained model
            with open(self.save_file, 'wb') as outfile:
                pickle.dump([features, labels], outfile)
            
            print(f"HOG detector initialized with {len(pieces_hog)} pieces and {len(squares_hog)} empty squares")
            return True
        except Exception as e:
            print(f"Error initializing HOG detector: {e}")
            traceback.print_exc()
            return False

    def load_hog(self):
        """Load previously saved HOG detector model"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'rb') as infile:
                    self.features, self.labels = pickle.load(infile)
                
                if len(self.features) > 0 and len(self.labels) > 0:
                    self.knn.train(self.features, cv2.ml.ROW_SAMPLE, self.labels)
                    print("HOG detector loaded successfully")
                    return True
                else:
                    print("Invalid HOG model data")
                    return False
            else:
                print("HOG model file not found. You need to play at least 1 game before starting a game from position.")
                return False
        except Exception as e:
            print(f"Error loading HOG detector: {e}")
            traceback.print_exc()
            return False

    def detect_state_cnn(self, chessboard_image):
        """Detect board state using CNN models"""
        # Check if result is already in cache
        image_hash = hash(chessboard_image.tobytes())
        if image_hash in self.state_cache:
            return self.state_cache[image_hash]
            
        # If CNN models are not available, return empty result
        if self.piece_model is None or self.color_model is None:
            return None
            
        try:
            state = []
            for row in range(8):
                row_state = []
                for column in range(8):
                    height, width = chessboard_image.shape[:2]
                    minX = int(column * width / 8)
                    maxX = int((column + 1) * width / 8)
                    minY = int(row * height / 8)
                    maxY = int((row + 1) * height / 8)
                    square_image = chessboard_image[minY:maxY, minX:maxX]
                    
                    # Skip processing if square is too small
                    if square_image.shape[0] < 10 or square_image.shape[1] < 10:
                        row_state.append('.')
                        continue
                    
                    is_piece = predict(square_image, self.piece_model)
                    if is_piece:
                        is_white = predict(square_image, self.color_model)
                        if is_white:
                            row_state.append('w')
                        else:
                            row_state.append('b')
                    else:
                        row_state.append('.')
                state.append(row_state)
            
            # Cache the result
            self.state_cache[image_hash] = state
            
            return state
        except Exception as e:
            print(f"Error detecting state with CNN: {e}")
            return None

    def check_state_cnn(self, result):
        """Check if CNN-detected state matches the current board state"""
        if result is None:
            return False
            
        try:
            for row in range(8):
                for column in range(8):
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    expected_state = '.'
                    if piece:
                        if piece.color == chess.WHITE:
                            expected_state = 'w'
                        else:
                            expected_state = 'b'

                    if result[row][column] != expected_state:
                        return False
            return True
        except Exception as e:
            print(f"Error checking CNN state: {e}")
            return False

    def detect_state_hog(self, chessboard_image):
        """Detect board state using HOG features and KNN classifier"""
        # Check if result is already in cache
        image_hash = hash(chessboard_image.tobytes())
        if image_hash in self.state_cache:
            return self.state_cache[image_hash]
            
        try:
            # Convert to grayscale if needed
            if len(chessboard_image.shape) == 3:
                chessboard_image_gray = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
            else:
                chessboard_image_gray = chessboard_image
            
            # Get HOG features for each square
            board_hog = []
            for row in range(8):
                hog_row = []
                for column in range(8):
                    square_img = get_square_image(row, column, chessboard_image_gray)
                    if square_img.shape[0] > 0 and square_img.shape[1] > 0:  # Ensure valid image
                        resized_square = cv2.resize(square_img, (64, 64))
                        hog_features = self.hog.compute(resized_square)
                        hog_row.append(hog_features)
                    else:
                        # Handle invalid squares (should not happen)
                        hog_row.append(np.zeros((3780, 1), np.float32))
                board_hog.append(hog_row)
            
            # Classify each square using KNN
            knn_result = []
            for row in range(8):
                knn_row = []
                for column in range(8):
                    # Find nearest neighbors
                    ret, result, neighbours, dist = self.knn.findNearest(np.array([board_hog[row][column]]), k=3)
                    knn_row.append(result[0][0])
                knn_result.append(knn_row)
            
            # Convert to boolean array (True for piece, False for empty)
            board_state = [[knn_result[row][column] > HOG_MATCH_THRESHOLD for column in range(8)] for row in range(8)]
            
            # Cache the result
            self.state_cache[image_hash] = board_state
            
            return board_state
        except Exception as e:
            print(f"Error detecting state with HOG: {e}")
            traceback.print_exc()
            return None

    def check_state_hog(self, result):
        """Check if HOG-detected state matches the current board state"""
        if result is None:
            return False
            
        try:
            for row in range(8):
                for column in range(8):
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    
                    # Check for mismatches
                    if piece and (not result[row][column]):
                        print(f"Expected piece at {square_name}")
                        return False
                    if (not piece) and result[row][column]:
                        print(f"Expected empty at {square_name}")
                        return False
            return True
        except Exception as e:
            print(f"Error checking HOG state: {e}")
            return False

    def check_state_for_move(self, result):
        """Check if Canny edge detection state matches the current board state"""
        if result is None:
            return False
            
        try:
            for row in range(8):
                for column in range(8):
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    
                    # Check for mismatches
                    if piece and (True not in result[row][column]):
                        print(f"Expected piece at {square_name}")
                        return False
                    if (not piece) and (False not in result[row][column]):
                        print(f"Expected empty at {square_name}")
                        return False
            return True
        except Exception as e:
            print(f"Error checking edge detection state: {e}")
            return False

    def check_state_for_light(self, result, result_hog):
        """Check for light changes using combined methods"""
        if result is None or result_hog is None:
            return False
            
        try:
            for row in range(8):
                for column in range(8):
                    # Use HOG result as fallback for ambiguous Canny results
                    if len(result[row][column]) > 1:
                        result[row][column] = [result_hog[row][column]]
                        
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    
                    # Check for mismatches
                    if piece and (False in result[row][column]):
                        print(f"Light change mismatch at {square_name}")
                        return False
                    if (not piece) and (True in result[row][column]):
                        print(f"Light change mismatch at {square_name}")
                        return False
            return True
        except Exception as e:
            print(f"Error checking for light changes: {e}")
            return False

    def get_valid_2_move_cnn(self, frame):
        """Try to detect two consecutive moves using CNN"""
        try:
            # Detect board state using CNN
            board_result = self.detect_state_cnn(frame)
            
            # Check for registered moves from commentator
            move_to_register = self.get_move_to_register()

            if move_to_register:
                # Apply the first registered move
                self.board.push(move_to_register)
                
                # Try all legal follow-up moves
                for move in self.board.legal_moves:
                    # Skip non-queen promotions
                    if move.promotion and move.promotion != chess.QUEEN:
                        continue
                    
                    # Apply the second move
                    self.board.push(move)
                    
                    # Check if resulting position matches the frame
                    if self.check_state_cnn(board_result):
                        # Found a match - undo second move
                        self.board.pop()
                        
                        # Get and announce move
                        valid_move_string = move_to_register.uci()
                        self.speech_thread.put_text(valid_move_string[:4])
                        
                        # Record the move
                        self.played_moves.append(move_to_register)
                        self.board.pop()  # Undo first move
                        self.executed_moves.append(self.board.san(move_to_register))
                        self.board.push(move_to_register)  # Reapply first move
                        
                        # Update turn
                        if hasattr(self.internet_game, 'is_our_turn'):
                            self.internet_game.is_our_turn = not self.internet_game.is_our_turn
                            
                        print(f"Detected two moves: {valid_move_string} followed by {move.uci()}")
                        return True, move.uci()
                    else:
                        # No match - undo second move and try next one
                        self.board.pop()
                
                # No matching second move found - undo first move
                self.board.pop()

            return False, ""
        except Exception as e:
            print(f"Error in get_valid_2_move_cnn: {e}")
            # Restore board state if exception occurred
            while self.board.move_stack and self.board.move_stack[-1] != move_to_register:
                self.board.pop()
            return False, ""

    def get_valid_move_cnn(self, frame):
        """Try to detect a single move using CNN"""
        try:
            # Detect board state using CNN
            board_result = self.detect_state_cnn(frame)
            
            # Check for registered moves from commentator
            move_to_register = self.get_move_to_register()

            if move_to_register:
                # Apply the move
                self.board.push(move_to_register)
                
                # Check if resulting position matches the frame
                if self.check_state_cnn(board_result):
                    # Found a match
                    self.board.pop()  # Undo the move for now
                    return True, move_to_register.uci()
                else:
                    # No match
                    self.board.pop()  # Undo the move
                    return False, ""
            else:
                # Try all legal moves
                for move in self.board.legal_moves:
                    # Skip non-queen promotions
                    if move.promotion and move.promotion != chess.QUEEN:
                        continue
                    
                    # Apply the move
                    self.board.push(move)
                    
                    # Check if resulting position matches the frame
                    if self.check_state_cnn(board_result):
                        # Found a match
                        self.board.pop()  # Undo the move for now
                        return True, move.uci()
                    else:
                        # No match - undo and try next move
                        self.board.pop()
                        
            return False, ""
        except Exception as e:
            print(f"Error in get_valid_move_cnn: {e}")
            # Restore board state if exception occurred
            while self.board.move_stack:
                self.board.pop()
            return False, ""

    def get_valid_move_hog(self, fgmask, frame):
        """Try to detect a move using HOG features and foreground mask"""
        try:
            # Extract potential squares from foreground mask
            board = [[self.board_basics.get_square_image(row, column, fgmask).mean() for column in range(8)] for row in range(8)]
            potential_squares = []
            square_scores = {}
            
            # Find squares with significant changes
            for row in range(8):
                for column in range(8):
                    score = board[row][column]
                    if score < SQUARE_SCORE_THRESHOLD:
                        continue
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    potential_squares.append(square)
                    square_scores[square] = score

            # Get registered move if available
            move_to_register = self.get_move_to_register()
            potential_moves = []

            # Detect board state using HOG
            board_result = self.detect_state_hog(frame)
            
            if move_to_register:
                # Check if registered move matches potential squares
                if (move_to_register.from_square in potential_squares) and (move_to_register.to_square in potential_squares):
                    # Apply the move
                    self.board.push(move_to_register)
                    
                    # Check if resulting position matches
                    if self.check_state_hog(board_result):
                        # Found a match
                        self.board.pop()  # Undo for now
                        return True, move_to_register.uci()
                    else:
                        # No match
                        self.board.pop()
                        return False, ""
            else:
                # Try all legal moves that involve the potential squares
                for move in self.board.legal_moves:
                    if (move.from_square in potential_squares) and (move.to_square in potential_squares):
                        # Skip non-queen promotions
                        if move.promotion and move.promotion != chess.QUEEN:
                            continue
                        
                        # Apply the move
                        self.board.push(move)
                        
                        # Check if resulting position matches
                        if self.check_state_hog(board_result):
                            # Found a match
                            self.board.pop()  # Undo for now
                            
                            # Calculate score based on square changes
                            total_score = square_scores[move.from_square] + square_scores[move.to_square]
                            potential_moves.append((total_score, move.uci()))
                        else:
                            # No match - undo and try next move
                            self.board.pop()
                            
            # Return the highest scoring move if any
            if potential_moves:
                return True, max(potential_moves)[1]
            else:
                return False, ""
        except Exception as e:
            print(f"Error in get_valid_move_hog: {e}")
            # Restore board state if exception occurred
            while self.board.move_stack:
                self.board.pop()
            return False, ""

    def get_valid_move_canny(self, fgmask, frame):
        """Try to detect a move using Canny edge detection"""
        if self.roi_mask is None:
            return False, ""
            
        try:
            # Extract potential squares from foreground mask
            board = [[self.board_basics.get_square_image(row, column, fgmask).mean() for column in range(8)] for row in range(8)]
            potential_squares = []
            square_scores = {}
            
            # Find squares with significant changes
            for row in range(8):
                for column in range(8):
                    score = board[row][column]
                    if score < SQUARE_SCORE_THRESHOLD:
                        continue
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    potential_squares.append(square)
                    square_scores[square] = score

            # Get registered move if available
            move_to_register = self.get_move_to_register()
            potential_moves = []

            # Detect board state using Canny edge detection
            board_result = detect_state(frame, self.board_basics.d[0], self.roi_mask)
            
            if move_to_register:
                # Check if registered move matches potential squares
                if (move_to_register.from_square in potential_squares) and (move_to_register.to_square in potential_squares):
                    # Apply the move
                    self.board.push(move_to_register)
                    
                    # Check if resulting position matches
                    if self.check_state_for_move(board_result):
                        # Found a match
                        self.board.pop()  # Undo for now
                        return True, move_to_register.uci()
                    else:
                        # No match
                        self.board.pop()
                        return False, ""
            else:
                # Try all legal moves that involve the potential squares
                for move in self.board.legal_moves:
                    if (move.from_square in potential_squares) and (move.to_square in potential_squares):
                        # Skip non-queen promotions
                        if move.promotion and move.promotion != chess.QUEEN:
                            continue
                        
                        # Apply the move
                        self.board.push(move)
                        
                        # Check if resulting position matches
                        if self.check_state_for_move(board_result):
                            # Found a match
                            self.board.pop()  # Undo for now
                            
                            # Calculate score based on square changes
                            total_score = square_scores[move.from_square] + square_scores[move.to_square]
                            potential_moves.append((total_score, move.uci()))
                        else:
                            # No match - undo and try next move
                            self.board.pop()
                            
            # Return the highest scoring move if any
            if potential_moves:
                return True, max(potential_moves)[1]
            else:
                return False, ""
        except Exception as e:
            print(f"Error in get_valid_move_canny: {e}")
            # Restore board state if exception occurred
            while self.board.move_stack:
                self.board.pop()
            return False, ""

    def get_move_to_register(self):
        """Get the next move to register from the commentator"""
        if self.commentator:
            try:
                if hasattr(self.commentator.game_state, 'registered_moves'):
                    if len(self.executed_moves) < len(self.commentator.game_state.registered_moves):
                        return self.commentator.game_state.registered_moves[len(self.executed_moves)]
            except (AttributeError, IndexError) as e:
                print(f"Error getting registered move: {e}")
        return None

    def is_light_change(self, frame):
        """Check if there's a change in lighting conditions rather than a move"""
        try:
            # First check using edge detection if ROI mask is available
            if self.roi_mask is not None:
                result = detect_state(frame, self.board_basics.d[0], self.roi_mask)
                result_hog = self.detect_state_hog(frame)
                state = self.check_state_for_light(result, result_hog)
                
                if state:
                    print("Light change detected (edge detection)")
                    return True
            
            # Then check using CNN
            result_cnn = self.detect_state_cnn(frame)
            state_cnn = self.check_state_cnn(result_cnn)
            
            if state_cnn:
                print("Light change detected (CNN)")
                return True
                
            return False
        except Exception as e:
            print(f"Error checking for light changes: {e}")
            return False

    def get_valid_move(self, potential_squares, potential_moves):
        """Try to find a valid move from potential squares and moves"""
        try:
            print("Potential squares:")
            print(potential_squares)
            print("Potential moves:")
            print(potential_moves)

            # Get registered move if available
            move_to_register = self.get_move_to_register()

            valid_move_string = ""
            for score, start, arrival in potential_moves:
                if valid_move_string:
                    break

                # Check if this move matches the registered move
                if move_to_register:
                    if chess.square_name(move_to_register.from_square) != start:
                        continue
                    if chess.square_name(move_to_register.to_square) != arrival:
                        continue

                # Create move in UCI format
                uci_move = start + arrival
                
                try:
                    move = chess.Move.from_uci(uci_move)
                except Exception as e:
                    print(f"Invalid UCI move: {uci_move} - {e}")
                    continue

                # Check if move is legal
                if move in self.board.legal_moves:
                    valid_move_string = uci_move
                else:
                    # Check for promotion
                    if move_to_register:
                        uci_move_promoted = move_to_register.uci()
                    else:
                        uci_move_promoted = uci_move + 'q'
                        
                    try:
                        promoted_move = chess.Move.from_uci(uci_move_promoted)
                        if promoted_move in self.board.legal_moves:
                            valid_move_string = uci_move_promoted
                    except Exception as e:
                        print(f"Invalid promotion move: {uci_move_promoted} - {e}")

            # Extract just the square names from potential squares
            square_names = [square[1] for square in potential_squares]
            print(f"Potential square names: {square_names}")
            
            # Check for castling (these patterns require special detection)
            castling_patterns = [
                # Format: [(square_list), legal_move, description]
                (["e1", "h1", "f1", "g1"], "e1g1", "White kingside castling"),
                (["e1", "a1", "c1", "d1"], "e1c1", "White queenside castling"),
                (["e8", "h8", "f8", "g8"], "e8g8", "Black kingside castling"),
                (["e8", "a8", "c8", "d8"], "e8c8", "Black queenside castling")
            ]
            
            for squares, move_str, description in castling_patterns:
                # Check if all required squares for this castling pattern are detected
                if all(square in square_names for square in squares):
                    try:
                        castling_move = chess.Move.from_uci(move_str)
                        if castling_move in self.board.legal_moves:
                            print(f"Detected {description}")
                            valid_move_string = move_str
                            break
                    except Exception as e:
                        print(f"Error processing castling move: {e}")

            # If we have a registered move but it doesn't match what we found
            if move_to_register and valid_move_string and (move_to_register.uci() != valid_move_string):
                print(f"Warning: Registered move {move_to_register.uci()} doesn't match detected move {valid_move_string}")
                return False, valid_move_string

            # Return the valid move if found
            if valid_move_string:
                return True, valid_move_string
            else:
                return False, ""
        except Exception as e:
            print(f"Error in get_valid_move: {e}")
            traceback.print_exc()
            return False, ""

    def register_move(self, fgmask, previous_frame, next_frame):
        """Main method to detect and register a move"""
        # Clear the state cache for new frames
        self.state_cache = {}
        
        try:
            # Try multiple detection methods in order of reliability
            detection_methods = [
                # (method_function, method_name)
                (lambda: self.get_valid_2_move_cnn(next_frame), "Two-move CNN detection"),
                (lambda: self.get_valid_move_cnn(next_frame), "CNN detection"),
                (lambda: self.get_valid_move(*self.board_basics.get_potential_moves(fgmask, previous_frame, next_frame, self.board)), "SSIM detection"),
                (lambda: self.get_valid_move_canny(fgmask, next_frame), "Canny edge detection"),
                (lambda: self.get_valid_move_hog(fgmask, next_frame), "HOG detection")
            ]
            
            # Try each method until one succeeds
            for method_func, method_name in detection_methods:
                success, valid_move_string = method_func()
                if success:
                    print(f"Move detected using {method_name}: {valid_move_string}")
                    break
            
            # If no method succeeded
            if not success:
                self.speech_thread.put_text(self.language.move_failed)
                print(f"Failed to detect move. Current position: {self.board.fen()}")
                return False
            
            # Parse the UCI move
            valid_move_UCI = chess.Move.from_uci(valid_move_string)

            print(f"Move registered: {valid_move_string} ({self.board.san(valid_move_UCI)})")

            # Execute the move on the online game if it's our turn or we're making opponent moves
            if hasattr(self.internet_game, 'is_our_turn') and (self.internet_game.is_our_turn or self.make_opponent):
                self.internet_game.move(valid_move_UCI)
                self.played_moves.append(valid_move_UCI)
                
                # Wait for commentator to register the move if available
                wait_counter = 0
                while self.commentator and wait_counter < 30:  # 3 second timeout
                    time.sleep(0.1)
                    wait_counter += 1
                    move_to_register = self.get_move_to_register()
                    if move_to_register:
                        valid_move_UCI = move_to_register
                        break
            else:
                # Just announce the move locally
                self.speech_thread.put_text(valid_move_string[:4])
                self.played_moves.append(valid_move_UCI)

            # Record the move in standard algebraic notation
            self.executed_moves.append(self.board.san(valid_move_UCI))
            
            # Record if it's a capture for SSIM updates
            is_capture = self.board.is_capture(valid_move_UCI)
            color = int(self.board.turn)
            
            # Update the board state
            self.board.push(valid_move_UCI)

            # Update turn information
            if hasattr(self.internet_game, 'is_our_turn'):
                self.internet_game.is_our_turn = not self.internet_game.is_our_turn

            # Update ML models with the new position
            self.learn(next_frame)
            
            # Update SSIM data
            self.board_basics.update_ssim(previous_frame, next_frame, valid_move_UCI, is_capture, color)
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error registering move: {e}")
            traceback.print_exc()
            self.speech_thread.put_text(self.language.move_failed)
            return False

    def learn(self, frame):
        """Update machine learning models based on the current position"""
        try:
            # Detect current state using HOG
            result = self.detect_state_hog(frame)
            if result is None:
                print("Warning: Could not detect state for learning")
                return
                
            # Convert to grayscale for HOG features
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
                
            # Lists to store new training examples
            new_pieces = []
            new_squares = []

            # Scan the board for misclassifications
            for row in range(8):
                for column in range(8):
                    square_name = self.board_basics.convert_row_column_to_square_name(row, column)
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    
                    # Get square image
                    square_img = get_square_image(row, column, frame_gray)
                    
                    # Check for misclassifications
                    if piece and (not result[row][column]):
                        print(f"Learning piece at {square_name}")
                        piece_hog = self.hog.compute(cv2.resize(square_img, (64, 64)))
                        new_pieces.append(piece_hog)
                    elif (not piece) and result[row][column]:
                        print(f"Learning empty at {square_name}")
                        square_hog = self.hog.compute(cv2.resize(square_img, (64, 64)))
                        new_squares.append(square_hog)
            
            # Skip if no new examples
            if not new_pieces and not new_squares:
                return
                
            # Create labels
            labels_pieces = np.ones((len(new_pieces), 1), np.int32)
            labels_squares = np.zeros((len(new_squares), 1), np.int32)
            
            # Add new pieces to training data
            if new_pieces:
                new_pieces = np.array(new_pieces)
                self.features = np.float32(np.concatenate((self.features, new_pieces), axis=0))
                self.labels = np.concatenate((self.labels, labels_pieces), axis=0)
                
            # Add new empty squares to training data
            if new_squares:
                new_squares = np.array(new_squares)
                self.features = np.float32(np.concatenate((self.features, new_squares), axis=0))
                self.labels = np.concatenate((self.labels, labels_squares), axis=0)

            # Limit dataset size to prevent memory issues (keep only the most recent examples)
            max_examples = 100
            if len(self.features) > max_examples:
                self.features = self.features[-max_examples:]
                self.labels = self.labels[-max_examples:]
            
            # Retrain the KNN classifier
            self.knn = cv2.ml.KNearest_create()
            self.knn.train(self.features, cv2.ml.ROW_SAMPLE, self.labels)
            
            # Save updated model
            with open(self.save_file, 'wb') as outfile:
                pickle.dump([self.features, self.labels], outfile)
                
            print(f"ML model updated with {len(new_pieces)} new pieces and {len(new_squares)} new empty squares")
            
        except Exception as e:
            print(f"Error updating ML model: {e}")
            traceback.print_exc()
            # Continue without updating the model if an error occurs
