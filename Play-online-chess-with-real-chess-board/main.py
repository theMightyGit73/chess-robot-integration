import time
import cv2
import pickle
import numpy as np
import sys
import os
import signal
import gc
import traceback
from collections import deque
import platform
from threading import Thread
from board_calibration_machine_learning import detect_board
from game import Game
from board_basics import Board_basics
from helper import perspective_transform
from speech import Speech_thread
from videocapture import Video_capture_thread
from languages import *

# Global constants for motion detection - tuned for better performance on Raspberry Pi
MOTION_START_THRESHOLD = 0.75  # Lowered for more sensitive motion detection
HISTORY = 80  # Reduced history for better responsiveness
MAX_MOVE_MEAN = 50
COUNTER_MAX_VALUE = 2  # Reduced to speed up motion detection

# Global variables for resource management
video_capture_thread = None
speech_thread = None
game = None
cleanup_done = False

def signal_handler(sig, frame):
    """Handle exit signals properly to ensure clean shutdown"""
    print("\nReceived termination signal. Cleaning up...")
    cleanup_resources()
    sys.exit(0)

def cleanup_resources():
    """Clean up all resources on exit"""
    global cleanup_done, video_capture_thread, speech_thread, game
    
    if not cleanup_done:
        print("Cleaning up resources...")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Stop threads
        if video_capture_thread is not None:
            try:
                video_capture_thread.stop()
                print("Camera resources released")
            except Exception as e:
                print(f"Error stopping video thread: {e}")
        
        if speech_thread is not None:
            try:
                speech_thread.stop_speaking = True
                print("Speech resources released")
            except Exception as e:
                print(f"Error stopping speech thread: {e}")
        
        # Force garbage collection
        gc.collect()
        
        cleanup_done = True
        print("Cleanup complete")

def parse_arguments():
    """Parse command line arguments into a settings dictionary"""
    settings = {
        'webcam_width': None,
        'webcam_height': None,
        'fps': None,
        'use_template': True,
        'make_opponent': False,
        'drag_drop': False,
        'comment_me': False,
        'comment_opponent': False,
        'calibrate': False,
        'start_delay': 5,  # seconds
        'cap_index': 0,
        'cap_api': cv2.CAP_ANY,
        'voice_index': 0,
        'language': English(),
        'token': "",
        'debug': False
    }
    
    for argument in sys.argv:
        if argument == "no-template":
            settings['use_template'] = False
        elif argument == "make-opponent":
            settings['make_opponent'] = True
        elif argument == "comment-me":
            settings['comment_me'] = True
        elif argument == "comment-opponent":
            settings['comment_opponent'] = True
        elif argument.startswith("delay="):
            settings['start_delay'] = int("".join(c for c in argument if c.isdigit()))
        elif argument == "drag":
            settings['drag_drop'] = True
        elif argument == "debug":
            settings['debug'] = True
        elif argument.startswith("cap="):
            settings['cap_index'] = int("".join(c for c in argument if c.isdigit()))
            platform_name = platform.system()
            if platform_name == "Darwin":
                settings['cap_api'] = cv2.CAP_AVFOUNDATION
            elif platform_name == "Linux":
                settings['cap_api'] = cv2.CAP_V4L2
            else:
                settings['cap_api'] = cv2.CAP_DSHOW
        elif argument.startswith("voice="):
            settings['voice_index'] = int("".join(c for c in argument if c.isdigit()))
        elif argument.startswith("lang="):
            if "German" in argument:
                settings['language'] = German()
            elif "Russian" in argument:
                settings['language'] = Russian()
            elif "Turkish" in argument:
                settings['language'] = Turkish()
            elif "Italian" in argument:
                settings['language'] = Italian()
            elif "French" in argument:
                settings['language'] = French()
        elif argument.startswith("token="):
            settings['token'] = argument[len("token="):].strip()
        elif argument == "calibrate":
            settings['calibrate'] = True
        elif argument.startswith("width="):
            settings['webcam_width'] = int(argument[len("width="):])
        elif argument.startswith("height="):
            settings['webcam_height'] = int(argument[len("height="):])
        elif argument.startswith("fps="):
            settings['fps'] = int(argument[len("fps="):])
    
    return settings

def init_camera():
    """Initialize and test the camera"""
    global video_capture_thread
    
    # Initialize video capture with Pi Camera
    video_capture_thread = Video_capture_thread()
    video_capture_thread.daemon = True

    # Start the camera and wait for initialization
    print("Initializing camera...")
    video_capture_thread.start()
    time.sleep(3)  # Give camera time to initialize

    # Test camera connection with multiple attempts
    for attempt in range(3):
        test_frame = video_capture_thread.get_frame()
        if test_frame is not None:
            print(f"Camera initialized with frame size: {test_frame.shape}")
            return True
        else:
            print(f"Camera initialization attempt {attempt+1}/3 failed. Retrying...")
            time.sleep(1)
    
    print("ERROR: Couldn't get frame from camera after multiple attempts.")
    print("Please check your camera connection and permissions.")
    return False

def calibrate_board(settings):
    """Calibrate the chess board using machine learning"""
    print("Starting board calibration...")
    
    try:
        # Load models
        corner_model = cv2.dnn.readNetFromONNX("yolo_corner.onnx")
        piece_model = cv2.dnn.readNetFromONNX("cnn_piece.onnx")
        color_model = cv2.dnn.readNetFromONNX("cnn_color.onnx")
        
        is_detected = False
        for attempt in range(100):
            print(f"Calibration attempt {attempt+1}/100...")
            frame = video_capture_thread.get_frame()
            if frame is None:
                print("Error reading frame. Please check your camera connection.")
                time.sleep(0.5)
                continue
                
            # Try to detect the board
            result = detect_board(frame, corner_model, piece_model, color_model)
            if result:
                pts1, side_view_compensation, rotation_count = result
                roi_mask = None
                is_detected = True
                print("Chess board detected successfully!")
                
                # Save calibration data
                filename = 'constants.bin'
                outfile = open(filename, 'wb')
                pickle.dump([True, [pts1, side_view_compensation, rotation_count]], outfile)
                outfile.close()
                print(f"Calibration data saved to {filename}")
                
                return pts1, side_view_compensation, rotation_count, roi_mask
        
        if not is_detected:
            print("Could not detect the chess board after multiple attempts.")
            print("Please check your lighting conditions and camera positioning.")
            print("Tips:")
            print("- Ensure there is even lighting without glare or shadows")
            print("- Position the camera directly above the board")
            print("- Make sure all four corners of the board are visible")
            return None
            
    except Exception as e:
        print(f"Calibration error: {e}")
        traceback.print_exc()
        return None

def load_calibration_data():
    """Load previously saved calibration data"""
    print("Loading calibration data from file...")
    try:
        filename = 'constants.bin'
        if not os.path.exists(filename):
            print(f"Error: Calibration file '{filename}' not found.")
            print("Please run board calibration first.")
            return None
            
        infile = open(filename, 'rb')
        calibration_data = pickle.load(infile)
        infile.close()
        
        if calibration_data[0]:
            pts1, side_view_compensation, rotation_count = calibration_data[1]
            roi_mask = None
            print("Loaded ML-based calibration data")
            print(f"  - Side view compensation: {side_view_compensation}")
            print(f"  - Rotation count: {rotation_count}")
        else:
            corners, side_view_compensation, rotation_count, roi_mask = calibration_data[1]
            pts1 = np.float32([list(corners[0][0]), list(corners[8][0]), list(corners[0][8]),
                              list(corners[8][8])])
            print("Loaded traditional calibration data")
            print(f"  - Side view compensation: {side_view_compensation}")
            print(f"  - Rotation count: {rotation_count}")
        
        return pts1, side_view_compensation, rotation_count, roi_mask
        
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        traceback.print_exc()
        print("Please run board calibration first.")
        return None

def waitUntilMotionCompletes(pts1, motion_fgbg):
    """Wait until there is no more motion on the board, with improved error handling"""
    counter = 0
    timeout_counter = 0
    max_timeout = 30  # Maximum seconds to wait for motion to complete
    
    start_time = time.time()
    
    while counter < COUNTER_MAX_VALUE:
        # Check for timeout
        if time.time() - start_time > max_timeout:
            print("Motion detection timeout - assuming motion has stopped")
            break
            
        frame = video_capture_thread.get_frame()
        if frame is None:
            print("Error reading frame during motion detection")
            timeout_counter += 1
            if timeout_counter > 5:
                print("Too many frame errors during motion detection")
                break
            time.sleep(0.5)
            continue
            
        try:
            # Transform and process the frame
            frame = perspective_transform(frame, pts1)
            fgmask = motion_fgbg.apply(frame)
            ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
            mean = fgmask.mean()
            
            if mean < MOTION_START_THRESHOLD:
                counter += 1
                if settings['debug']:
                    print(f"Motion settling: {counter}/{COUNTER_MAX_VALUE}, mean={mean:.2f}")
            else:
                counter = 0
                if settings['debug'] and counter % 5 == 0:
                    print(f"Motion detected: mean={mean:.2f}")
        except Exception as e:
            print(f"Error in motion detection: {e}")
            time.sleep(0.5)
    
    return True

def stabilize_background_subtractors(pts1, move_fgbg, motion_fgbg, settings):
    """Stabilize the background subtractor models with improved error handling"""
    print("Stabilizing background subtractors...")
    
    # First phase - stabilize motion detection
    best_mean = float("inf")
    counter = 0
    timeout_counter = 0
    
    start_time = time.time()
    
    while counter < COUNTER_MAX_VALUE:
        # Check for timeout
        if time.time() - start_time > 10:  # 10 second timeout
            print("Stabilization timeout - continuing with current state")
            break
            
        frame = video_capture_thread.get_frame()
        if frame is None:
            print("Error reading frame during background stabilization")
            timeout_counter += 1
            if timeout_counter > 5:
                print("Too many frame errors during stabilization")
                break
            time.sleep(0.5)
            continue
            
        try:
            frame = perspective_transform(frame, pts1)
            move_fgbg.apply(frame)
            fgmask = motion_fgbg.apply(frame, learningRate=0.1)
            ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
            mean = fgmask.mean()
            
            if settings['debug'] and counter % 5 == 0:
                print(f"Motion stabilization: mean={mean:.2f}, best={best_mean:.2f}")
            
            if mean >= best_mean:
                counter += 1
            else:
                best_mean = mean
                counter = 0
        except Exception as e:
            print(f"Error in motion stabilization: {e}")
            time.sleep(0.5)

    # Second phase - stabilize move detection
    best_mean = float("inf")
    counter = 0
    timeout_counter = 0
    
    start_time = time.time()
    
    while counter < COUNTER_MAX_VALUE:
        # Check for timeout
        if time.time() - start_time > 10:  # 10 second timeout
            print("Move stabilization timeout - continuing with current state")
            break
            
        frame = video_capture_thread.get_frame()
        if frame is None:
            print("Error reading frame during move stabilization")
            timeout_counter += 1
            if timeout_counter > 5:
                print("Too many frame errors during move stabilization")
                break
            time.sleep(0.5)
            continue
            
        try:
            frame = perspective_transform(frame, pts1)
            fgmask = move_fgbg.apply(frame, learningRate=0.1)
            ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
            motion_fgbg.apply(frame)
            mean = fgmask.mean()
            
            if settings['debug'] and counter % 5 == 0:
                print(f"Move stabilization: mean={mean:.2f}, best={best_mean:.2f}")
            
            if mean >= best_mean:
                counter += 1
            else:
                best_mean = mean
                counter = 0
        except Exception as e:
            print(f"Error in move stabilization: {e}")
            time.sleep(0.5)

    # Get one more frame for the initial state
    for _ in range(5):  # Try up to 5 times
        frame = video_capture_thread.get_frame()
        if frame is not None:
            try:
                return perspective_transform(frame, pts1)
            except Exception as e:
                print(f"Error transforming final frame: {e}")
                time.sleep(0.5)
    
    print("WARNING: Could not get a valid frame after stabilization")
    return None

def show_diagnostic_view(frame, fgmask, mean, settings):
    """Show diagnostic view if debug mode is enabled"""
    if not settings['debug']:
        return
        
    try:
        # Create a diagnostic display
        h, w = frame.shape[:2]
        
        # Resize mask for display
        display_mask = cv2.resize(fgmask, (w, h))
        
        # Create a side-by-side view
        diagnostic = np.hstack((frame, cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)))
        
        # Add text with motion value
        cv2.putText(
            diagnostic,
            f"Motion: {mean:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0) if mean > MOTION_START_THRESHOLD else (0, 0, 255), 
            2
        )
        
        # Show the diagnostic view
        cv2.imshow("Diagnostic", diagnostic)
        cv2.waitKey(1)
    except Exception as e:
        print(f"Error showing diagnostic view: {e}")

def check_system_status():
    """Check system status periodically"""
    try:
        # Check CPU temperature on Raspberry Pi
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                if temp > 80:
                    print(f"WARNING: CPU temperature is very high: {temp:.1f}°C")
                    print("System may throttle or become unstable")
                elif temp > 70:
                    print(f"Note: CPU temperature is elevated: {temp:.1f}°C")
        
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            print(f"WARNING: Memory usage is very high: {memory.percent}%")
            print("System may become unstable")
            # Force garbage collection
            gc.collect()
    except ImportError:
        # psutil not available
        pass
    except Exception as e:
        print(f"Error checking system status: {e}")

def run_game_loop(settings, pts1, move_fgbg, motion_fgbg, board_basics, game):
    """Main game loop with improved error handling and performance optimizations"""
    print("Stabilizing background subtractors...")
    previous_frame = stabilize_background_subtractors(pts1, move_fgbg, motion_fgbg, settings)
    if previous_frame is None:
        print("ERROR: Failed to get initial board state")
        return False
        
    previous_frame_queue = deque(maxlen=10)
    previous_frame_queue.append(previous_frame)

    # Start the game
    # Start the game
    print("Starting game...")

    # Initialize speech if available
    try:
        speech_thread.put_text(settings['language'].game_started)
    except Exception as e:
        print(f"Speech error: {e}")
        print("Continuing without speech...")

    # Check if commentator exists before trying to start it
    if hasattr(game, 'commentator') and game.commentator is not None:
        game.commentator.start()
    else:
        # If no commentator available, create a dummy one or handle the situation
        print("Warning: No commentator available. Game will continue without move announcements.")
        
        # Create a dummy game state that won't break the code
        class DummyGameState:
            def __init__(self):
                self.variant = 'standard'
                self.resign_or_draw = False
                self.registered_moves = []  # Add this to prevent attribute errors
        
        # Assign a minimal placeholder to avoid NoneType errors
        if not hasattr(game, 'commentator') or game.commentator is None:
            from commentator import Commentator_thread
            game.commentator = Commentator_thread()
            game.commentator.game_state = DummyGameState()
            
    # Wait for game state to be initialized
    print("Waiting for game state...")
    wait_start = time.time()
    while game.commentator.game_state.variant == 'wait':
        if time.time() - wait_start > 30:  # 30 second timeout
            print("Timeout waiting for game state - continuing with standard mode")
            break
        time.sleep(0.1)

    # Initialize SSIM or HOG based on game variant
    try:
        if game.commentator.game_state.variant == 'standard':
            print("Initializing standard game...")
            board_basics.initialize_ssim(previous_frame)
            game.initialize_hog(previous_frame)
        else:
            print("Loading saved game data...")
            board_basics.load_ssim()
            game.load_hog()
    except Exception as e:
        print(f"Error initializing game: {e}")
        traceback.print_exc()
        return False

    print("\n=== Game is ready ===")
    print("Make your moves on the physical board.")
    
    # Frame processing counter for periodic tasks
    frame_counter = 0
    status_check_interval = 50  # Check system status every 50 frames
    
    # Performance monitoring variables
    start_time = time.time()
    processed_frames = 0
    
    error_count = 0  # Count errors to detect persistent issues
    
    # Main game loop
    while not game.board.is_game_over() and not game.commentator.game_state.resign_or_draw:
        try:
            # Periodic tasks
            frame_counter += 1
            
            # Periodically check system status
            if frame_counter % status_check_interval == 0:
                check_system_status()
                
                # Log performance statistics
                elapsed = time.time() - start_time
                if elapsed > 0 and processed_frames > 0:
                    fps = processed_frames / elapsed
                    print(f"Performance: {fps:.1f} frames/second")
                    # Reset counters
                    start_time = time.time()
                    processed_frames = 0
                    
                # Force garbage collection periodically
                gc.collect()
            
            sys.stdout.flush()
            
            # Get current frame
            frame = video_capture_thread.get_frame()
            if frame is None:
                print("Error reading frame during game")
                time.sleep(0.5)
                error_count += 1
                if error_count > 10:
                    print("Too many consecutive frame errors - check camera connection")
                    return False
                continue
            
            # Reset error counter since we got a valid frame
            error_count = 0
            processed_frames += 1
                
            # Process frame to detect motion
            try:
                frame = perspective_transform(frame, pts1)
                fgmask = motion_fgbg.apply(frame)
                ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
                kernel = np.ones((11, 11), np.uint8)
                fgmask = cv2.erode(fgmask, kernel, iterations=1)
                mean = fgmask.mean()
                
                # Show diagnostic view if enabled
                show_diagnostic_view(frame, fgmask, mean, settings)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
            
            # Check if there is significant motion (a move is being made)
            if mean > MOTION_START_THRESHOLD:
                print("Motion detected - waiting for move to complete...")
                
                # Wait until the move is completed
                waitUntilMotionCompletes(pts1, motion_fgbg)
                
                print("Motion completed - analyzing move...")
                
                # Capture the new board state
                frame = video_capture_thread.get_frame()
                if frame is None:
                    print("Error reading frame after move completion")
                    time.sleep(0.5)
                    continue
                    
                try:
                    frame = perspective_transform(frame, pts1)
                    fgmask = move_fgbg.apply(frame, learningRate=0.0)
                    
                    # Threshold the mask if necessary
                    if fgmask.mean() >= 10.0:
                        ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
                        
                    # Reset mask if it's too noisy
                    if fgmask.mean() >= MAX_MOVE_MEAN:
                        print("Mask is too noisy - resetting")
                        fgmask = np.zeros(fgmask.shape, dtype=np.uint8)
                        
                    # Update background models
                    motion_fgbg.apply(frame)
                    move_fgbg.apply(frame, learningRate=1.0)
                    
                    # Stabilize and get last frame
                    last_frame = stabilize_background_subtractors(pts1, move_fgbg, motion_fgbg, settings)
                    if last_frame is None:
                        print("Failed to get stable frame after move - using current frame")
                        last_frame = frame
                        
                    previous_frame = previous_frame_queue[0]

                    # Try to register the move
                    if (not game.is_light_change(last_frame)) and game.register_move(fgmask, previous_frame, last_frame):
                        print("Move registered successfully")
                    else:
                        print("Move registration failed - please redo your move")
                        
                    # Update frame queue
                    previous_frame_queue = deque(maxlen=10)
                    previous_frame_queue.append(last_frame)
                    
                except Exception as e:
                    print(f"Error processing move: {e}")
                    traceback.print_exc()
            else:
                # No significant motion, update background model and queue
                try:
                    move_fgbg.apply(frame)
                    
                    # Only add to queue every few frames to reduce memory usage
                    if frame_counter % 3 == 0:
                        previous_frame_queue.append(frame)
                except Exception as e:
                    print(f"Error updating background model: {e}")
        
        except Exception as e:
            print(f"Error in main game loop: {e}")
            traceback.print_exc()
            time.sleep(1)
            error_count += 1
            if error_count > 10:
                print("Too many errors in main loop - exiting")
                return False

    # Game completed
    print("\n=== Game completed ===")
    if game.board.is_game_over():
        result = game.board.result()
        print(f"Game result: {result}")
        
        # Announce winner if possible
        if result == "1-0":
            print("White wins!")
        elif result == "0-1":
            print("Black wins!")
        elif result == "1/2-1/2":
            print("Draw!")
    else:
        print("Game ended by resignation or draw offer")
    
    return True

# Main program
if __name__ == "__main__":
    try:
        # Register signal handlers for clean exit
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Parse command line arguments
        settings = parse_arguments()
        
        print("\n=== Chess Recognition System ===")
        if settings['debug']:
            print("Debug mode enabled - additional information will be displayed")
        
        # Initialize camera
        if not init_camera():
            sys.exit(1)
        
        # Get calibration data (either by calibrating or loading)
        if settings['calibrate']:
            calibration_result = calibrate_board(settings)
        else:
            calibration_result = load_calibration_data()
            
        if calibration_result is None:
            cleanup_resources()
            sys.exit(1)
        
        pts1, side_view_compensation, rotation_count, roi_mask = calibration_result
        
        # Initialize the board basics with the calibration data
        board_basics = Board_basics(side_view_compensation, rotation_count)

        # Initialize speech thread
        speech_thread = Speech_thread()
        speech_thread.daemon = True
        speech_thread.index = settings['voice_index']
        
        # Try to start speech thread, but continue even if it fails
        try:
            speech_thread.start()
        except Exception as e:
            print(f"Warning: Speech initialization failed: {e}")
            print("Continuing without speech support...")

        # Initialize game
        game = Game(
            board_basics, 
            speech_thread, 
            settings['use_template'], 
            settings['make_opponent'], 
            settings['start_delay'], 
            settings['comment_me'], 
            settings['comment_opponent'],
            settings['drag_drop'], 
            settings['language'], 
            settings['token'], 
            roi_mask
        )
        
        # Create background subtractors
        move_fgbg = cv2.createBackgroundSubtractorKNN()
        motion_fgbg = cv2.createBackgroundSubtractorKNN(history=HISTORY)
        
        # Run main game loop
        success = run_game_loop(settings, pts1, move_fgbg, motion_fgbg, board_basics, game)
        
        # Clean up
        cleanup_resources()
        
        if success:
            print("Program completed successfully")
            sys.exit(0)
        else:
            print("Program completed with errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()
        cleanup_resources()
        sys.exit(1)
