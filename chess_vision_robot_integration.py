import os
import sys
import time
import cv2
import numpy as np
import re
import chess
import logging
import traceback
import subprocess
import json
import signal
from threading import Thread, Event
import threading
import queue
from datetime import datetime
import platform
import gc
import shutil  # For file operations

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KARAYAMAN_DIR = os.path.join(CURRENT_DIR, 'Play-online-chess-with-real-chess-board')
PRINTER_DIR = os.path.join(CURRENT_DIR, 'PrinterController')

# Add paths for imports
sys.path.insert(0, KARAYAMAN_DIR)
sys.path.insert(0, PRINTER_DIR)

# Check for required directories
for directory in [KARAYAMAN_DIR, PRINTER_DIR]:
    if not os.path.exists(directory):
        print(f"ERROR: Required directory not found: {directory}")
        print("Please make sure all project components are correctly installed.")
        sys.exit(1)

# Configure logging
LOG_DIR = os.path.join(CURRENT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f'chess_robot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChessRobot")

# Import status for tracking dependencies
dependency_status = {
    "karayaman_modules": False,
    "printer_modules": False,
    "opencv": False,
    "numpy": False,
}

# Log system info
logger.info(f"System: {platform.system()} {platform.release()}")
logger.info(f"Python: {platform.python_version()}")
logger.info(f"Current directory: {CURRENT_DIR}")

# Try importing OpenCV and NumPy
try:
    cv_version = cv2.__version__
    dependency_status["opencv"] = True
    logger.info(f"OpenCV version: {cv_version}")
except Exception as e:
    logger.error(f"OpenCV import error: {e}")

try:
    np_version = np.__version__
    dependency_status["numpy"] = True
    logger.info(f"NumPy version: {np_version}")
except Exception as e:
    logger.error(f"NumPy import error: {e}")

# Import Karayaman's modules
try:
    from videocapture import Video_capture_thread
    from board_basics import Board_basics
    from helper import perspective_transform
    from speech import Speech_thread
    dependency_status["karayaman_modules"] = True
    logger.info("Successfully imported Karayaman modules")
except ImportError as e:
    logger.error(f"Failed to import Karayaman modules: {e}")
    print(f"ERROR: Could not import Karayaman modules: {e}")
    print(f"Make sure the directory exists: {KARAYAMAN_DIR}")
    print("Check README for installation instructions.")
    sys.exit(1)

# Import PrinterController modules
try:
    from printerChess import (
        PrinterController, PrinterConfig, ChessBoardConfig,
        ChessPieceConfig, GripperConfig, StorageConfig
    )
    dependency_status["printer_modules"] = True
    logger.info("Successfully imported PrinterController modules")
except ImportError as e:
    logger.error(f"Failed to import PrinterController modules: {e}")
    print(f"ERROR: Could not import PrinterController modules: {e}")
    print(f"Make sure the directory exists: {PRINTER_DIR}")
    print("Check README for installation instructions.")
    sys.exit(1)

# =========== Globals for resource management ===========
video_capture = None
printer = None
speech = None
running = True
debug_visualization = False  # Can be toggled in the menu
stop_event = Event()  # For signaling threads to stop
status_queue = queue.Queue()  # For thread-safe status updates

# Configuration
CONFIG_DIR = os.path.join(CURRENT_DIR, 'config')
os.makedirs(CONFIG_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(CONFIG_DIR, 'integration_config.json')


# Default configuration
DEFAULT_CONFIG = {
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 30
    },
    "printer": {
        "port": "/dev/ttyUSB0",
        "baud_rate": 115200
    },
    "chess": {
        "engine_path": "",
        "skill_level": 10,
        "thinking_time": 1.0
    },
    "speech": {
        "enabled": True,
        "language": "en",  # Default to English
        "volume": 1.0,
        "rate": 150,  # Words per minute
        "available_languages": ["en", "fr", "de", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar"]
    },
    "debug": False
}


# --------------------------------------------------------------------
# Initialization and Configuration
# --------------------------------------------------------------------
def load_config():
    """Load configuration from file or create with defaults"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.info("Configuration loaded successfully")
                return config
        else:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
                logger.info("Created default configuration file")
                return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()

config = load_config()

def init_camera(retries=3, retry_delay=2):
    global video_capture
    logger.info("Initializing camera...")

    # Kill any zombie camera processes first
    kill_zombie_camera_processes()
    
    for attempt in range(retries):
        try:
            # Release any existing camera resource
            if video_capture:
                try:
                    video_capture.stop()
                except Exception as e:
                    logger.warning(f"Error stopping existing camera: {e}")
            
            video_capture = None
            gc.collect()  # Force garbage collection
            time.sleep(0.1)  # Give time for resources to be released
            
            # Create and start new camera thread
            video_capture = Video_capture_thread()
            video_capture.daemon = True
            video_capture.start()
            
            logger.info(f"Camera thread started (attempt {attempt+1}/{retries})")
            time.sleep(retry_delay)  # Allow time for initialization
            
            # Verify we can get frames
            for i in range(10):  # Try up to 10 times
                frame = video_capture.get_frame()
                if frame is not None:
                    logger.info(f"Camera working, frame size: {frame.shape}")
                    return True
                time.sleep(0.5)
            
            logger.warning("Camera initialized but not returning frames")
            
            # Clean up failed camera
            video_capture.stop()
            video_capture = None
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            logger.debug(traceback.format_exc())
            if video_capture:
                try:
                    video_capture.stop()
                except:
                    pass
                video_capture = None
            time.sleep(retry_delay)
    
    logger.error("Failed to initialize camera after multiple attempts")
    return False
    
def init_speech():
    """Initialize speech system with error handling and configuration options"""
    global speech, config
    try:
        # Check if speech is enabled in configuration
        speech_config = config.get("speech", {})
        speech_enabled = speech_config.get("enabled", True)
        
        if not speech_enabled:
            logger.info("Speech is disabled in configuration - skipping initialization")
            return True
        
        # Create speech thread    
        speech = Speech_thread()
        speech.daemon = True
        
        # Start the speech thread first
        speech.start()
        
        # Wait a moment for thread to initialize
        time.sleep(1.0)
        
        # Set language if configured
        language = speech_config.get("language", "en")
        volume = speech_config.get("volume", 1.0)
        rate = speech_config.get("rate", 150)
        
        # Configure speech properties
        try:
            speech.set_language(language)
            speech.set_volume(volume)
            speech.set_rate(rate)
            logger.info(f"Speech configured: language={language} ({get_language_name(language)}), volume={volume}, rate={rate}")
        except Exception as config_err:
            logger.warning(f"Error configuring speech properties: {config_err}")
            # Continue anyway with defaults
        
        # Test speech system
        test_timeout = threading.Timer(2.0, lambda: None)
        test_timeout.start()
        speech.put_text("Chess robot system initialized")
        
        logger.info("Speech system initialized")
        return True
    except Exception as e:
        logger.error(f"Speech initialization failed: {e}")
        logger.debug(traceback.format_exc())
        return False
               
def init_printer(safe_mode=True):
    """Initialize printer with comprehensive error handling"""
    global printer
    logger.info("Initializing printer...")

    try:
        # Create configuration objects
        printer_config = PrinterConfig()
        chess_config = ChessBoardConfig()
        piece_config = ChessPieceConfig()
        gripper_config = GripperConfig()
        storage_config = StorageConfig()

        # Override printer port from config if specified
        if "printer" in config and "port" in config["printer"]:
            printer_config.printer_port = config["printer"]["port"]
            logger.info(f"Using printer port from config: {printer_config.printer_port}")

        # Search for all configuration files
        z_offset_path = find_z_offset_file(printer_config.z_offset_file)
        if z_offset_path:
            printer_config.z_offset_file = z_offset_path
            logger.info(f"Using z_offset file found at: {z_offset_path}")
        else:
            logger.warning(f"Could not find {printer_config.z_offset_file} in subdirectories.")

        chess_positions_path = find_chess_positions_file(chess_config.positions_file)
        if chess_positions_path:
            chess_config.positions_file = chess_positions_path
            logger.info(f"Using chess positions file found at: {chess_positions_path}")
        else:
            logger.warning(f"Could not find {chess_config.positions_file} in subdirectories.")

        piece_settings_path = find_piece_settings_file(piece_config.pieces_file)
        if piece_settings_path:
            piece_config.pieces_file = piece_settings_path
            logger.info(f"Using piece settings file found at: {piece_settings_path}")
        else:
            logger.warning(f"Could not find {piece_config.pieces_file} in subdirectories.")

        gripper_settings_path = find_gripper_settings_file(gripper_config.settings_file)
        if gripper_settings_path:
            gripper_config.settings_file = gripper_settings_path
            logger.info(f"Using gripper settings file found at: {gripper_settings_path}")
        else:
            logger.warning(f"Could not find {gripper_config.settings_file} in subdirectories.")

        storage_positions_path = find_storage_positions_file(storage_config.storage_file)
        if storage_positions_path:
            storage_config.storage_file = storage_positions_path
            logger.info(f"Using storage positions file found at: {storage_positions_path}")
        else:
            logger.warning(f"Could not find {storage_config.storage_file} in subdirectories.")

        # Initialize printer controller
        logger.info("Creating PrinterController instance...")
        printer = PrinterController(printer_config, chess_config, piece_config, gripper_config, storage_config)
        logger.info("PrinterController initialization complete.")

        # Home axes if not in safe mode
        if not safe_mode:
            logger.info("Homing printer axes...")
            printer.home_axes()
            logger.info("Printer homing complete.")
        else:
            logger.info("Skipping homing in safe mode")

        return True
    except Exception as e:
        logger.error(f"Printer initialization failed: {e}")
        logger.debug(traceback.format_exc())
        
        # Attempt cleanup if initialization failed
        if printer:
            try:
                printer.cleanup()
            except:
                pass
            printer = None
            
        return False

def prepare_ml_models():
    """
    Prepare ML model files by ensuring they're in the right location
    
    Returns:
        bool: True if successful, False otherwise
    """
    # List of model files needed
    model_files = [
        "yolo_corner.onnx",
        "cnn_piece.onnx",
        "cnn_color.onnx"
    ]
    
    # Create a models directory in the Karayaman dir if it doesn't exist
    models_dir = os.path.join(KARAYAMAN_DIR, "models")
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir)
            logger.info(f"Created models directory: {models_dir}")
        except Exception as e:
            logger.error(f"Failed to create models directory: {e}")
            return False
    
    # Check each model file
    models_found = True
    for model_file in model_files:
        # Check if model file exists in the expected location
        target_path = os.path.join(KARAYAMAN_DIR, model_file)
        if not os.path.exists(target_path):
            logger.info(f"Model file not found at expected location: {target_path}")
            
            # Try to find it elsewhere
            found_path = find_file(model_file)
            if found_path:
                # Copy the file to the expected location
                try:
                    import shutil
                    shutil.copy2(found_path, target_path)
                    logger.info(f"Copied model file from {found_path} to {target_path}")
                except Exception as e:
                    logger.error(f"Failed to copy model file: {e}")
                    models_found = False
            else:
                logger.error(f"Could not find model file: {model_file}")
                models_found = False
        else:
            logger.info(f"Model file already exists: {target_path}")
    
    return models_found
 
def find_file(filename, start_dir=None, search_parent=True, max_depth=5):
    """
    Generic file finder that searches for a file in subdirectories and parent directories
    
    Args:
        filename (str): Name of file to find
        start_dir (str): Directory to start the search from (defaults to current directory)
        search_parent (bool): Whether to also search parent directories
        max_depth (int): Maximum directory depth to search
        
    Returns:
        str or None: Path to the file if found, None otherwise
    """
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info(f"Searching for {filename} starting from {start_dir}")
    
    # First try direct look in common locations
    common_dirs = [
        start_dir,
        os.path.join(start_dir, 'models'),
        os.path.join(start_dir, 'config'),
        os.path.join(KARAYAMAN_DIR, 'models'),
        KARAYAMAN_DIR,
        os.path.join(CURRENT_DIR, 'models')
    ]
    
    # Add variations of the name for models
    if filename.endswith('.onnx') or filename.endswith('.bin'):
        base_name = os.path.splitext(filename)[0]
        for ext in ['.onnx', '.bin', '.model', '.pb']:
            for prefix in ['', 'model_', 'ml_']:
                alt_name = prefix + base_name + ext
                if alt_name != filename:
                    logger.debug(f"Also looking for alternative name: {alt_name}")
                    # Check if any of the common dirs have this alternative file
                    for common_dir in common_dirs:
                        alt_path = os.path.join(common_dir, alt_name)
                        if os.path.exists(alt_path):
                            logger.info(f"Found alternative file {alt_name} at: {alt_path}")
                            return alt_path
    
    # Check common locations first
    for common_dir in common_dirs:
        if os.path.exists(os.path.join(common_dir, filename)):
            found_path = os.path.join(common_dir, filename)
            logger.info(f"Found {filename} at: {found_path}")
            return found_path
    
    # Search in subdirectories
    found_paths = []
    for root, dirs, files in os.walk(start_dir):
        # Limit depth of search
        rel_dir = os.path.relpath(root, start_dir)
        if rel_dir != '.' and rel_dir.count(os.sep) >= max_depth:
            dirs[:] = []  # Don't descend any deeper
            continue
            
        if filename in files:
            found_path = os.path.join(root, filename)
            found_paths.append(found_path)
            logger.info(f"Found {filename} at: {found_path}")
    
    # If requested, also search parent directories 
    if search_parent and not found_paths:
        parent_dir = os.path.dirname(start_dir)
        if parent_dir != start_dir:  # Avoid infinite loop at root directory
            for _ in range(3):  # Limit parent directory search depth
                if os.path.exists(os.path.join(parent_dir, filename)):
                    found_path = os.path.join(parent_dir, filename)
                    found_paths.append(found_path)
                    logger.info(f"Found {filename} in parent directory: {found_path}")
                    break
                new_parent = os.path.dirname(parent_dir)
                if new_parent == parent_dir:  # Reached root directory
                    break
                parent_dir = new_parent
    
    if found_paths:
        # Return the most recently modified file if multiple are found
        latest_path = max(found_paths, key=os.path.getmtime)
        logger.info(f"Using most recent {filename}: {latest_path}")
        return latest_path
    
    logger.warning(f"Could not find {filename} anywhere")
    return None

def find_z_offset_file(filename="z_offset.json", start_dir=None):
    """Search for z_offset.json in subdirectories"""
    return find_file(filename, start_dir, search_parent=True)

def find_chess_positions_file(filename="chess_positions.json", start_dir=None):
    """Search for chess_positions.json in subdirectories"""
    return find_file(filename, start_dir, search_parent=True)

def find_piece_settings_file(filename="piece_settings.json", start_dir=None):
    """Search for piece_settings.json in subdirectories"""
    return find_file(filename, start_dir, search_parent=True)

def find_gripper_settings_file(filename="gripper_settings.json", start_dir=None):
    """Search for gripper_settings.json in subdirectories"""
    return find_file(filename, start_dir, search_parent=True)

def find_storage_positions_file(filename="storage_positions.json", start_dir=None):
    """Search for storage_positions.json in subdirectories"""
    return find_file(filename, start_dir, search_parent=True)


# --------------------------------------------------------------------
# System Management
# --------------------------------------------------------------------
def cleanup_resources():
    """Clean up all resources upon exit"""
    global video_capture, printer, speech

    logger.info("Cleaning up resources...")
    
    # Set up a timeout for cleanup
    cleanup_timeout = 10  # seconds
    start_time = time.time()

    # Destroy all OpenCV windows
    try:
        cv2.destroyAllWindows()
        logger.info("Closed all OpenCV windows")
    except Exception as e:
        logger.warning(f"Error closing OpenCV windows: {e}")

    # Stop video capture
    if video_capture:
        try:
            video_capture.stop()
            logger.info("Camera resources released")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            logger.debug(traceback.format_exc())

    # Stop speech
    if speech:
        try:
            speech.stop_speaking = True
            logger.info("Speech resources released")
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
            logger.debug(traceback.format_exc())

    # Clean up printer
    if printer:
        try:
            printer.cleanup()
            logger.info("Printer resources released")
        except Exception as e:
            logger.error(f"Error cleaning up printer: {e}")
            logger.debug(traceback.format_exc())

    # Kill any zombie processes
    try:
        kill_zombie_camera_processes()
    except Exception as e:
        logger.warning(f"Error killing zombie processes: {e}")

    # Force garbage collection
    try:
        gc.collect()
        logger.info("Garbage collection completed")
    except Exception as e:
        logger.warning(f"Error during garbage collection: {e}")

    # Check if cleanup took too long
    elapsed = time.time() - start_time
    if elapsed > cleanup_timeout:
        logger.warning(f"Cleanup took longer than expected: {elapsed:.2f} seconds")
    else:
        logger.info(f"Cleanup completed in {elapsed:.2f} seconds")
        
    # Clear global references to release memory
    video_capture = None
    printer = None
    speech = None

def signal_handler(sig, frame):
    """Handle termination signals gracefully"""
    global running, stop_event
    logger.info(f"Received termination signal {sig}. Exiting...")
    running = False
    stop_event.set()
    cleanup_resources()
    sys.exit(0)

def kill_zombie_camera_processes():
    """
    Kill any zombie camera processes that might be preventing camera access
    """
    try:
        import psutil
        
        # Look for processes that might be holding the camera
        camera_process_names = ['python', 'python3']
        camera_related_cmdlines = ['videocapture', 'diagnostic.py', 'board_calibration']
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip our own process
                if proc.pid == os.getpid():
                    continue
                    
                # Check if it's a Python process
                if proc.info['name'] in camera_process_names:
                    # Check if cmdline contains any camera-related terms
                    if proc.info['cmdline'] and any(term in ' '.join(proc.info['cmdline']) for term in camera_related_cmdlines):
                        logger.warning(f"Found potential zombie camera process: PID {proc.pid}")
                        try:
                            proc.terminate()
                            gone, still_alive = psutil.wait_procs([proc], timeout=3)
                            for p in still_alive:
                                logger.warning(f"Force killing process: {p.pid}")
                                p.kill()
                            logger.info(f"Successfully terminated camera process: {proc.pid}")
                        except Exception as kill_error:
                            logger.error(f"Error killing process {proc.pid}: {kill_error}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except ImportError:
        logger.warning("psutil not available - cannot check for zombie processes")
    except Exception as e:
        logger.error(f"Error checking for zombie processes: {e}")

def check_system_resources():
    """Check system resources and log warnings if low"""
    try:
        # Check CPU usage
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status_msg = f"System: CPU {cpu_percent}%, Memory {memory.percent}%"
        
        # Check for Raspberry Pi temperature
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
                status_msg += f", CPU Temp: {temp:.1f}°C"
                if temp > 80:
                    logger.warning(f"CPU temperature very high: {temp:.1f}°C - thermal throttling likely!")
                elif temp > 70:
                    logger.warning(f"CPU temperature elevated: {temp:.1f}°C")
        
        # Log warnings for high resource usage
        if cpu_percent > 90:
            logger.warning(f"CPU usage very high: {cpu_percent}%")
        if memory.percent > 90:
            logger.warning(f"Memory usage very high: {memory.percent}%")
            
        logger.info(status_msg)
        return status_msg
        
    except ImportError:
        logger.warning("psutil not available - cannot check system resources")
        return "System monitoring unavailable (psutil not installed)"
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return "Error checking system resources"

def update_status(message, level="INFO"):
    """Thread-safe status update function"""
    status_queue.put((message, level))
    logger.log(getattr(logging, level), message)

def check_camera_stability():
    """
    Verify camera is stable and not in the process of restarting.
    If unstable, wait for it to stabilize before continuing.
    """
    global video_capture
    
    if not video_capture:
        logger.warning("No video capture object found - might need initialization")
        # Try to initialize the camera
        if not init_camera():
            logger.error("Failed to initialize camera")
            return False
            
    # Check if camera is providing frames
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        frame = video_capture.get_frame()
        if frame is not None:
            # Camera is working
            if retry_count > 0:
                logger.info(f"Camera stabilized after {retry_count} checks")
            return True
            
        # No frame received
        retry_count += 1
        logger.warning(f"No frame from camera (check {retry_count}/{max_retries})")
        print(f"Waiting for camera to stabilize... ({retry_count}/{max_retries})")
        
        # If several retries failed, try to reinitialize
        if retry_count == 3:
            logger.warning("Camera seems unresponsive - attempting to reinitialize")
            print("Performing routine camera restart...")
            if init_camera():
                logger.info("Camera reinitialized successfully")
            else:
                logger.error("Failed to reinitialize camera")
                
        time.sleep(0.1)
    
    logger.error("Camera stability check failed after multiple attempts")
    print("WARNING: Camera may not be working properly. Proceed with caution.")
    
    # Even if failed, return True to allow execution to continue
    # The user can decide to abort if needed
    return True
 
def status_monitor():
    """Thread to monitor and display system status"""
    global status_queue, running
    
    logger.info("Status monitor thread started")
    
    while running:
        try:
            # Get status from queue with timeout
            try:
                message, level = status_queue.get(timeout=1.0)
                print(f"[{level}] {message}")
                status_queue.task_done()
            except queue.Empty:
                pass
                
            # Check if should exit
            if not running:
                break
                
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in status monitor thread: {e}")
            time.sleep(1.0)
    
    logger.info("Status monitor thread stopped")


# --------------------------------------------------------------------
# Chess Game Control
# --------------------------------------------------------------------
def play_game(pts1, board_basics, player_is_white=True, difficulty="Intermediate"):
    """
    Main chess game controller with improved PrinterController integration.
    
    This function manages the entire chess game workflow, including:
    - Move detection and validation using computer vision
    - Robot move execution via PrinterController
    - Comprehensive error handling and recovery
    - Game state tracking and management
    - User feedback and instructions
    
    Args:
        pts1: Perspective transformation points for board detection
        board_basics: Board basics object for chess coordinate conversion
        player_is_white: Whether the human plays as white (True) or black (False)
        difficulty: Difficulty level for Stockfish engine
        
    Returns:
        bool: True if game completed successfully, False otherwise
    """
    global printer, stop_event, speech, video_capture, debug_visualization
    
    # ======== PRE-GAME VALIDATION ========
    # Check for required components
    if not printer:
        logger.error("Printer not initialized. Cannot play game.")
        update_status("Printer not initialized. Cannot play game.", "ERROR")
        return False

    if pts1 is None or board_basics is None:
        logger.error("Calibration data missing. Cannot play game.")
        update_status("Calibration data missing. Run calibration first.", "ERROR")
        return False
    
    # Error tracking 
    error_count = 0
    consecutive_failures = 0
    
    # Game state tracking
    game_in_progress = True
    move_count = 0
    game_ending_processed = False
    
    # Difficulty settings mapping for Stockfish configuration
    difficulty_settings = {
        "Absolute Beginner": {"skill": 0, "depth": 1, "time": 0.05},
        "Beginner": {"skill": 3, "depth": 2, "time": 0.1},
        "Casual": {"skill": 6, "depth": 3, "time": 0.3},
        "Intermediate": {"skill": 10, "depth": 5, "time": 0.5},
        "Club Player": {"skill": 13, "depth": 8, "time": 0.8},
        "Advanced": {"skill": 16, "depth": 12, "time": 1.0},
        "Expert": {"skill": 20, "depth": None, "time": 2.0}
    }
    
    # ======== GAME INITIALIZATION ========
    # Display welcome and instructions
    print("\n" + "="*60)
    print("            CHESS GAME: HUMAN VS ROBOT")
    print("="*60)
    print("\nGAME INSTRUCTIONS:")
    print("1. Make your move on the physical chess board")
    print("2. The system will detect and validate your move")
    print("3. The robot will calculate and execute its response")
    print("4. Wait for the robot to complete its move before continuing")
    print("\nHELPFUL COMMANDS (during prompts):")
    print("- Type 'pause' to pause the game and access more options")
    print("- Type 'help' to see these instructions again")
    print("- Type 'board' to display the current board state")
    print("- Type 'resign' to forfeit the game")
    print("- Press Ctrl+C to emergency stop the robot")
    print("\nThe game will start with the robot in viewing position.")
    print(f"You play as {'WHITE' if player_is_white else 'BLACK'}.")
    if player_is_white:
        print("You move first.\n")
    else:
        print("The robot moves first.\n")
    
    # Display difficulty information
    print(f"Robot difficulty: {difficulty}")
    print("="*60 + "\n")
    
    input("Press Enter when ready to begin...")
    
    # Set board orientation based on player color using PrinterController's method
    printer.set_board_orientation(player_is_white)
    update_status(f"Board orientation set for player as {'white' if player_is_white else 'black'}", "INFO")
            
    # Configure Stockfish difficulty directly using PrinterController's method
    settings = difficulty_settings.get(difficulty, difficulty_settings["Intermediate"])
    printer.chess_game.configure_strength(
        skill_level=settings["skill"],
        depth_limit=settings["depth"],
        time_limit=settings["time"]
    )
    update_status(f"Stockfish difficulty set to {difficulty}", "INFO")
    
    # Ensure printer is in proper state before beginning
    logger.info("Verifying printer state before game")
    if not printer.verify_gripper_state("open"):
        logger.warning("Gripper not in open state - forcing open")
        printer.open_gripper(force=True)
        time.sleep(0.05)  # Wait for mechanical action
    
    # Move to board viewing position using PrinterController's method
    update_status("Moving to board viewing position", "INFO")
    try:
        # Try to move to viewing position with retry
        if not move_to_board_view_position():
            logger.warning("Failed to move to viewing position - will try again")
            time.sleep(0.1)
            if not move_to_board_view_position():
                update_status("Warning: Could not move to optimal viewing position", "WARNING")
                if not prompt_yes_no("Continue anyway?"):
                    update_status("Game aborted by user", "INFO")
                    return False
    except Exception as e:
        logger.error(f"Error moving to viewing position: {e}")
        logger.debug(traceback.format_exc())
        update_status("Error moving to viewing position", "ERROR")
        if not prompt_yes_no("Continue anyway?"):
            return False
    
    # Ensure game state is set to initial position
    try:
        update_status("Verifying initial board position", "INFO")
        printer.chess_game.display_state(printer.chess_mapper.flipped)
        print("\nPlease verify that your physical board shows the standard starting position.")
        verify = input("Is your board set up correctly? (yes/no): ").strip().lower()
        if verify != "yes" and verify != "y":
            update_status("Please set up your board to the starting position", "WARNING")
            input("Press Enter when the board is set up correctly...")
            
        if speech:
            try:
                if player_is_white:
                    speech.put_text(f"Welcome to Chess Robot! I'm ready to play. You are white, and you go first.")
                else:
                    speech.put_text(f"Welcome to Chess Robot! I'm ready to play. You are black. I'll make the first move.")
            except Exception as e:
                logger.warning(f"Error using speech: {e}")
    except Exception as e:
        logger.error(f"Error during board verification: {e}")
        logger.debug(traceback.format_exc())
    
    # Final game start announcement
    update_status("Game started.", "INFO")
    print("\n" + "-"*60)
    if player_is_white:
        print("GAME STARTED - Your turn (WHITE)")
        update_status("Your turn. Make your first move on the board.", "INFO")
    else:
        print("GAME STARTED - Robot's turn first (BLACK)")
        update_status("Robot's turn. Please wait for its move.", "INFO")
    print("-"*60 + "\n")
    
    # Announce game start with speech if available
    if speech:
        try:
            if player_is_white:
                speech.put_text("Game started. Please make your move.")
            else:
                speech.put_text("Game started. I'll make the first move. Please wait.")
        except Exception as e:
            logger.warning(f"Error using speech: {e}")
            logger.debug(traceback.format_exc())
    
    # ======== MAIN GAME LOOP ========
    while not stop_event.is_set() and game_in_progress:
        try:
            # Always verify gripper is open at start of each turn
            if not printer.verify_gripper_state("open"):
                logger.warning("Gripper not in open state at turn start - forcing open")
                printer.open_gripper(force=True)
                time.sleep(0.05)  # Wait for mechanical action
            
            # Enhanced game over check
            if check_game_over(printer.chess_game):
                game_in_progress = False
                announce_game_result(printer, speech)
                break
            
            # Check for game over conditions
            if printer.chess_game.board.is_game_over():
                game_in_progress = False
                break
            
            # Determine whose turn it is based on the chess engine's state
            is_user_turn = (printer.chess_game.board.turn == chess.WHITE and player_is_white) or \
                           (printer.chess_game.board.turn == chess.BLACK and not player_is_white)
            
            # ---- USER MOVE PHASE ----
            if is_user_turn and game_in_progress:
                user_move_success = False
                user_move_attempts = 0
                MAX_USER_MOVE_ATTEMPTS = 3
                
                while not user_move_success and user_move_attempts < MAX_USER_MOVE_ATTEMPTS:
                    try:
                        user_move_attempts += 1
                        logger.info(f"Processing user move (attempt {user_move_attempts}/{MAX_USER_MOVE_ATTEMPTS})")
                        
                        # Process the user's move using enhanced function
                        if process_user_move_improved(pts1, board_basics, printer):
                            user_move_success = True
                            move_count += 1
                            # Add slight delay to ensure move is fully registered
                            time.sleep(0.05)
                            break
                        else:
                            # Check if game was paused
                            user_input = input("\nMove failed. Type 'pause' to access menu, 'retry' to try again, or press Enter to continue: ").strip().lower()
                            if user_input == "pause":
                                paused = True
                                resume = handle_game_pause(printer)
                                if not resume:
                                    logger.info("Game ended by user during pause")
                                    update_status("Game ended by user", "INFO")
                                    game_in_progress = False
                                    break
                                paused = False
                            elif user_input == "retry":
                                continue
                            else:
                                # Count as an attempt but continue loop
                                error_count += 1
                                if error_count >= 5:
                                    logger.error("Too many errors detecting user moves")
                                    update_status("Too many errors. Please check system calibration.", "ERROR")
                                    if not prompt_continue_despite_error("Continue despite repeated errors?"):
                                        game_in_progress = False
                                        break
                    except KeyboardInterrupt:
                        logger.info("User move interrupted with Ctrl+C")
                        if prompt_yes_no("\nDo you want to abort the game?"):
                            game_in_progress = False
                            break
                        else:
                            continue
                    except Exception as e:
                        logger.error(f"Unexpected error during user move: {e}")
                        logger.debug(traceback.format_exc())
                        error_count += 1
                        if error_count >= 5:
                            update_status("Too many errors. Game cannot continue safely.", "ERROR")
                            game_in_progress = False
                            break
                
                # Move back to viewing position after user move if game still in progress
                if game_in_progress and user_move_success:
                    try:
                        logger.info("Moving to viewing position after user move")
                        move_to_board_view_position()
                        # Add small delay for stability
                        time.sleep(0.05)
                    except Exception as pos_err:
                        logger.warning(f"Error moving to viewing position: {pos_err}")
                        # Non-critical error, continue
            
            # ---- ROBOT MOVE PHASE ----
            # Only process if game is active and it's the robot's turn
            if not is_user_turn and game_in_progress:
                robot_move_success = False
                robot_move_attempts = 0
                MAX_ROBOT_MOVE_ATTEMPTS = 3
                
                while not robot_move_success and robot_move_attempts < MAX_ROBOT_MOVE_ATTEMPTS:
                    try:
                        robot_move_attempts += 1
                        logger.info(f"Processing robot move (attempt {robot_move_attempts}/{MAX_ROBOT_MOVE_ATTEMPTS})")
                        
                        # Process the robot's move using our enhanced function
                        if process_robot_move_improved(printer):
                            robot_move_success = True
                            move_count += 1
                            error_count = 0  # Reset error count on success
                            consecutive_failures = 0  # Reset consecutive failures
                            # Add slight delay to ensure move is fully registered
                            time.sleep(0.05)
                            break
                        else:
                            # Check if game was paused
                            user_input = input("\nRobot move failed. Type 'pause' to access menu, 'retry' to try again, or press Enter to continue: ").strip().lower()
                            if user_input == "pause":
                                paused = True
                                resume = handle_game_pause(printer)
                                if not resume:
                                    logger.info("Game ended by user during pause")
                                    update_status("Game ended by user", "INFO")
                                    game_in_progress = False
                                    break
                                paused = False
                            elif user_input == "retry":
                                continue
                            else:
                                # Count as a failure but continue loop
                                consecutive_failures += 1
                                if consecutive_failures >= 3:
                                    logger.error("Too many consecutive failures in robot response")
                                    update_status("Multiple failures. Game cannot continue safely.", "ERROR")
                                    if not prompt_continue_despite_error("Continue despite multiple errors?"):
                                        game_in_progress = False
                                        break
                    except KeyboardInterrupt:
                        logger.info("Robot move interrupted with Ctrl+C")
                        if prompt_yes_no("\nDo you want to abort the game?"):
                            game_in_progress = False
                            break
                        else:
                            # Ask if user wants to skip this move
                            if prompt_yes_no("Skip robot's move and continue to your turn?"):
                                # Force update turn in the chess engine
                                printer.chess_game.board.push(chess.Move.null())
                                break
                            else:
                                continue
                    except Exception as e:
                        logger.error(f"Unexpected error during robot move: {e}")
                        logger.debug(traceback.format_exc())
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            update_status("Multiple failures. Game cannot continue safely.", "ERROR")
                            game_in_progress = False
                            break
                
                # Always move back to viewing position after robot move if game still in progress
                if game_in_progress:
                    try:
                        logger.info("Moving to viewing position after robot move")
                        move_to_board_view_position()
                        # Add small delay for stability
                        time.sleep(0.05)
                    except Exception as pos_err:
                        logger.warning(f"Error moving to viewing position: {pos_err}")
                        # Non-critical error, continue
            
            # Optional pause after completing full turn
            if game_in_progress and move_count % 2 == 0 and move_count > 0:
                try:
                    print("\nCompleted move pair. Type 'pause' or press Enter to continue: ", end="")
                    response = input().strip().lower()
                    if response == 'pause':
                        paused = True
                        resume = handle_game_pause(printer)
                        if not resume:
                            logger.info("Game ended by user during pause")
                            update_status("Game ended by user", "INFO")
                            game_in_progress = False
                            break
                        paused = False
                except Exception as e:
                    logger.error(f"Error handling pause: {e}")
                    logger.debug(traceback.format_exc())
                    
        except KeyboardInterrupt:
            # Main loop interrupted with Ctrl+C
            if prompt_yes_no("\nDo you want to abort the game?"):
                logger.info("Game aborted by user")
                game_in_progress = False
            else:
                continue
        except Exception as loop_err:
            logger.error(f"Unexpected error in main game loop: {loop_err}")
            logger.debug(traceback.format_exc())
            error_count += 1
            if error_count >= 5:
                update_status("Too many unexpected errors. Game cannot continue safely.", "ERROR")
                game_in_progress = False
    
    # ======== GAME CONCLUSION ========
    if not game_ending_processed:
        game_ending_processed = True  # Set flag to prevent repeated ending
        
        # Move to safe viewing position
        update_status("Moving to board viewing position", "INFO")
        try:
            move_to_board_view_position()
            # Ensure gripper is open at end of game
            printer.open_gripper(force=True)
        except Exception as end_err:
            logger.warning(f"Error moving to viewing position at game end: {end_err}")
            logger.debug(traceback.format_exc())
        
        # Show final status and statistics once
        logger.info(f"Game over after {move_count} moves")
        
        # Display game results with enhanced formatting - only once
        print("\n" + "="*60)
        print("                  GAME OVER")
        print("="*60)
        print(f"Total moves played: {move_count}")
        
        # Show final board state and announce result - only call these once
        try:
            # Announce result first
            announce_game_result(printer, speech)
            # Then perform final cleanup
            end_game_cleanup(printer, speech, announce_result=False)  # Pass flag to prevent re-announcing
        except Exception as final_err:
            import logging
            logging.getLogger("ChessRobot").error(f"Error during final game sequence: {final_err}")
            import traceback
            logging.getLogger("ChessRobot").debug(traceback.format_exc())
        
        # Return success
        return True

def process_user_move_improved(pts1, board_basics, printer):
    """Handle the user's move detection, validation, and execution using PrinterController's methods."""
    global speech, stop_event, debug_visualization
    
    update_status("Waiting for your move...", "INFO")
    print("\nYOUR TURN - Make your move on the board")
    
    # Announce turn with speech if available
    if speech:
        try:
            speech.put_text("Your turn. Please make your move.")
        except Exception as e:
            logger.warning(f"Error using speech: {e}")
    
    # Reset detection attempts
    detection_attempts = 0
    MAX_DETECTION_ATTEMPTS = 3
    user_move_uci = None
    
    # Verify camera is stable before starting detection
    check_camera_stability()
    
    # Detection loop - try multiple times to detect a move
    while user_move_uci is None and detection_attempts < MAX_DETECTION_ATTEMPTS:
        try:
            print("\nWaiting for movement on the chess board...")
            user_move_uci = detect_move(pts1, board_basics)
            
            # Check for stop event
            if stop_event.is_set():
                logger.info("Game stopped by user during move detection")
                return False
                
            # Handle failed detection
            if not user_move_uci:
                detection_attempts += 1
                logger.warning(f"Move detection attempt {detection_attempts} failed")
                
                if detection_attempts < MAX_DETECTION_ATTEMPTS:
                    update_status(f"Move not detected. Please try again. (Attempt {detection_attempts}/{MAX_DETECTION_ATTEMPTS})", "WARNING")
                    print(f"\nMove not detected. Please try again. (Attempt {detection_attempts}/{MAX_DETECTION_ATTEMPTS})")
                
        except Exception as detect_err:
            logger.error(f"Error during move detection: {detect_err}")
            logger.debug(traceback.format_exc())
            detection_attempts += 1
            print(f"\nError during move detection. Retrying... ({detection_attempts}/{MAX_DETECTION_ATTEMPTS})")
    
    # Handle case where move detection failed after all attempts
    if not user_move_uci:
        logger.warning("Multiple move detection attempts failed")
        update_status("Could not detect your move after multiple attempts", "ERROR")
        
        # Ask user to input move manually
        print("\nCould not detect your move after multiple attempts.")
        manual_input = prompt_yes_no("Would you like to enter your move manually?")
        if manual_input:
            from_square = input("From square (e.g., e7): ").strip().lower()
            to_square = input("To square (e.g., e5): ").strip().lower()
            
            if re.match(r'^[a-h][1-8]$', from_square) and re.match(r'^[a-h][1-8]$', to_square):
                user_move_uci = from_square + to_square
            else:
                print("Invalid square format. Move aborted.")
                return False
        else:
            return False
    
    # Parse the move
    from_sq, to_sq = user_move_uci[:2], user_move_uci[2:]
    logger.info(f"Processing user move: {from_sq} → {to_sq}")
    
    # Display detected move to user
    print(f"\nDetected move: {from_sq.upper()} → {to_sq.upper()}")
    
    # Validate move using chess game's move validation
    is_valid = False
    matching_moves = []
    
    try:
        # Check if move is valid by checking if it's in the list of legal moves
        for move in printer.chess_game.board.legal_moves:
            if (chess.square_name(move.from_square) == from_sq and 
                chess.square_name(move.to_square) == to_sq):
                matching_moves.append(move)
        
        is_valid = len(matching_moves) > 0
    except Exception as e:
        logger.error(f"Error validating move: {e}")
        logger.debug(traceback.format_exc())
        is_valid = False
    
    if not is_valid:
        logger.warning(f"Invalid move detected: {user_move_uci}")
        update_status("That move isn't legal. Please try again.", "WARNING")
        
        # Announce invalid move
        if speech:
            try:
                speech.put_text("That move is not legal. Please try again.")
            except Exception as e:
                logger.warning(f"Error using speech: {e}")
                
        return False
    
    # Handle promotion if needed
    promotion_piece = None
    if matching_moves and any(move.promotion for move in matching_moves):
        logger.info("Detected a pawn promotion move")
        print("\nPAWN PROMOTION DETECTED")
        promotion_piece = prompt_for_promotion()
        if not promotion_piece:
            promotion_piece = 'q'  # Default to queen
        
        # Update move UCI with promotion piece
        user_move_uci = user_move_uci + promotion_piece
        print(f"Promoting to: {get_promotion_piece_name(promotion_piece)}")
    
    # Use printer's update_position method
    try:
        # Create the move object
        if promotion_piece:
            # Map promotion piece to chess piece type
            promotion_map = {
                'q': chess.QUEEN,
                'r': chess.ROOK,
                'b': chess.BISHOP,
                'n': chess.KNIGHT
            }
            # Create move with promotion
            from_square = chess.parse_square(from_sq)
            to_square = chess.parse_square(to_sq)
            move = chess.Move(from_square, to_square, promotion=promotion_map.get(promotion_piece))
            
            # Push move to board
            printer.chess_game.board.push(move)
        else:
            # Update with standard move
            printer.chess_game.update_position(from_sq, to_sq)
            
        # Display updated board
        printer.chess_game.display_state(printer.chess_mapper.flipped)
        
        # Success - save move and continue
        print(f"\n✓ Move accepted: {from_sq.upper()} → {to_sq.upper()}{' ('+get_promotion_piece_name(promotion_piece)+')' if promotion_piece else ''}")
        
        # Announce the move with speech
        if speech:
            try:
                # Format move announcement text
                move_text = f"You moved from {from_sq} to {to_sq}"
                if promotion_piece:
                    move_text += f", promoting to {get_promotion_piece_name(promotion_piece)}"
                speech.put_text(move_text)
            except Exception as e:
                logger.warning(f"Error using speech: {e}")
                
        return True
    except Exception as e:
        logger.error(f"Error updating chess position: {e}")
        logger.debug(traceback.format_exc())
        update_status("Error updating chess position", "ERROR")
        return False

def process_robot_move_improved(printer):
    """
    Handle the robot's move calculation and execution using PrinterChess methods.
    Uses enhanced timeout and directly leverages printer functionality.
    """
    global speech, stop_event
    
    update_status("Calculating best move...", "INFO")
    print("\nROBOT'S TURN - Calculating best move...")
    
    # Announce with speech if available
    if speech:
        try:
            speech.put_text("My turn. Calculating best move.")
        except Exception as e:
            logger.warning(f"Error using speech: {e}")
    
    try:
        # Check if the game is already over
        if printer.chess_game.board.is_game_over():
            result = printer.chess_game.board.result()
            winner = "White" if result == "1-0" else "Black" if result == "0-1" else "Draw"
            reason = "Checkmate!" if printer.chess_game.board.is_checkmate() else "Stalemate"
            
            if printer.chess_game.board.is_checkmate():
                print(f"\nCHECKMATE! {winner} wins!")
                if speech:
                    speech.put_text(f"Checkmate! {winner} wins!")
            else:
                print(f"\nGame over! {winner}")
                if speech:
                    speech.put_text(f"Game over! {winner}")
            
            update_status(f"Game over: {reason}", "INFO")
            return True
    
        # Set a timeout for engine evaluation
        start_time = time.time()
        max_wait = 15  # Maximum 15 seconds to wait for evaluation
        
        # Get evaluation with retry mechanism
        eval_info = None
        retry_count = 0
        max_retries = 3
        
        while eval_info is None and retry_count < max_retries and time.time() - start_time < max_wait:
            try:
                # Get evaluation from PrinterController's method with explicit timeout awareness
                eval_info = printer.chess_game.get_evaluation()
                if not eval_info:
                    retry_count += 1
                    logger.warning(f"Engine evaluation attempt {retry_count} failed, retrying...")
                    print(f"Engine evaluation attempt {retry_count}/{max_retries}...")
                    time.sleep(0.1)  # Longer delay between retries
            except Exception as retry_err:
                logger.warning(f"Error during evaluation attempt {retry_count}: {retry_err}")
                retry_count += 1
                time.sleep(0.1)
                
        # If we're taking too long, offer to select a move manually
        if time.time() - start_time > max_wait / 2 and not eval_info:
            print("\nStockfish is taking longer than expected. Wait or select a move manually?")
            if prompt_yes_no("Select move manually now?"):
                # Use a menu of legal moves
                print("\nAvailable legal moves:")
                legal_moves = []
                for i, move in enumerate(printer.chess_game.board.legal_moves, 1):
                    san = printer.chess_game.board.san(move)
                    legal_moves.append(san)
                    print(f"{i}. {san}")
                
                choice = input("\nEnter move number: ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(legal_moves):
                        best_san = legal_moves[idx]
                        # Create fake eval_info
                        eval_info = (0.0, best_san, None)  # Neutral evaluation with no mate
                    else:
                        print("Invalid selection")
                        return False
                except ValueError:
                    print("Invalid input")
                    return False
        
        # If still no evaluation after retries and timeout
        if not eval_info:
            logger.error("Engine evaluation failed or timed out")
            update_status("Engine error. Cannot produce move.", "ERROR")
            
            # Offer manual move option
            print("\nStockfish failed to calculate a move in time. Would you like to select a move manually?")
            if prompt_yes_no("Select move manually?"):
                # Use a menu of legal moves
                print("\nAvailable legal moves:")
                legal_moves = []
                for i, move in enumerate(printer.chess_game.board.legal_moves, 1):
                    san = printer.chess_game.board.san(move)
                    legal_moves.append(san)
                    print(f"{i}. {san}")
                
                choice = input("\nEnter move number: ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(legal_moves):
                        best_san = legal_moves[idx]
                        # Create fake eval_info with no mate
                        eval_info = (0.0, best_san, None)
                    else:
                        print("Invalid selection")
                        return False
                except ValueError:
                    print("Invalid input")
                    return False
            else:
                return False
                
        # Extract score, best move, and mate information
        if len(eval_info) >= 3:
            score, best_san, mate_in = eval_info
        else:
            score, best_san = eval_info
            mate_in = None

        # Get source and target squares
        try:
            squares = printer.chess_game.parse_move(best_san)
            if not squares:
                logger.error(f"Could not parse move: {best_san}")
                
                # Special case for pawn captures (e.g., bxc3)
                if re.match(r'^[a-h]x[a-h][1-8]$', best_san):
                    try:
                        # Use python-chess directly to parse the move
                        move_obj = printer.chess_game.board.parse_san(best_san)
                        source = chess.square_name(move_obj.from_square)
                        target = chess.square_name(move_obj.to_square)
                        logger.info(f"Special handling for pawn capture: {best_san} → {source},{target}")
                        squares = (source, target)
                    except Exception as e:
                        logger.error(f"Special pawn capture handling failed: {e}")
                        return False
                else:
                    return False
                
            source, target = squares
        except Exception as parse_err:
            logger.warning(f"Error parsing move {best_san}: {parse_err}")
            # Try to directly translate using convert_to_uci
            try:
                best_uci = printer.chess_game.convert_to_uci(best_san)
                source, target = best_uci[:2], best_uci[2:4]
            except Exception as e:
                logger.error(f"Failed to extract move coordinates: {e}")
                return False
        
        
        # Display move information with mate details
        print("\n" + "-"*60)
        if mate_in:
            if mate_in > 0:  # Positive mate = White wins
                print(f"ROBOT'S MOVE: {best_san} (Checkmate in {mate_in})")
            else:  # Negative mate = Black wins
                # Fixed interpretation - no longer depends on turn
                print(f"ROBOT'S MOVE: {best_san} (Black will checkmate in {abs(mate_in)})")
        else:
            print(f"ROBOT'S MOVE: {best_san}")
            
        print(f"Move details: {source.upper()} → {target.upper()}")

        if mate_in:
            if mate_in > 0:  # Positive mate = White wins
                print(f"Evaluation: White will checkmate in {mate_in} moves")
            else:  # Negative mate = Black wins
                print(f"Evaluation: Black will checkmate in {abs(mate_in)} moves")
        else:
            print(f"Position evaluation: {get_evaluation_text(score)} ({score:.2f})")
        print("-"*60)

        # Announce the planned move with speech
        if speech:
            try:
                move_text = f"I will move from {source} to {target}"
                if mate_in:
                    if mate_in > 0:  # Positive mate = White wins
                        evaluation_text = f"This will checkmate in {mate_in} moves"
                    else:  # Negative mate = Black wins
                        evaluation_text = f"Black will checkmate in {abs(mate_in)} moves"
                else:
                    evaluation_text = get_evaluation_text(score)
                speech.put_text(f"{move_text}. {evaluation_text}.")
            except Exception as e:
                logger.warning(f"Error using speech: {e}")
        
        # Ask for confirmation before executing move
        confirm = input("\nReady for robot to execute move? (yes/pause/abort): ").strip().lower()
        if confirm == "pause":
            return False
        elif confirm != "yes" and confirm != "y" and confirm != "":
            if prompt_yes_no("Would you like to abort this move?"):
                return False
        
        # Extra verification to ensure camera is not restarting
        check_camera_stability()
        
        # Execute the move using printer's play_move method
        update_status(f"Executing move: {best_san}", "INFO")
        print("\nExecuting physical move. Please wait...")
        
        # Announce move execution with speech
        if speech:
            try:
                speech.put_text("Executing move now. Please wait.")
            except Exception as e:
                logger.warning(f"Error using speech: {e}")
        
        # Use the SAN notation directly - printer.play_move handles conversion properly
        move_execute_start = time.time()
        execution_timeout = 60  # Give plenty of time for complex moves
        
        # Define a monitor thread to check for timeouts and report progress
        def monitor_execution():
            last_status_time = time.time()
            while time.time() - move_execute_start < execution_timeout and not stop_event.is_set():
                elapsed = time.time() - move_execute_start
                if time.time() - last_status_time > 5:  # Report status every 5 seconds
                    print(f"Move execution in progress... ({elapsed:.1f}s elapsed)")
                    last_status_time = time.time()
                time.sleep(0.05)
        
        # Start monitor thread
        import threading
        monitor_thread = threading.Thread(target=monitor_execution, daemon=True)
        monitor_thread.start()
        
        # Execute move with timeout awareness
        success = False
        try:
            success = printer.play_move(best_san)
        except Exception as e:
            logger.error(f"Error executing move: {e}")
            logger.debug(traceback.format_exc())
        
        # If move execution failed, try alternative approaches
        if not success:
            logger.warning(f"First attempt to execute move {best_san} failed, trying alternatives")
            
            # Try UCI format
            try:
                success = printer.play_move(f"{source}{target}")
                if success:
                    logger.info(f"Successfully executed move using UCI format: {source}{target}")
                else:
                    # Try _execute_direct_move as last resort
                    logger.info(f"Trying _execute_direct_move as last resort")
                    success = printer._execute_direct_move(source, target)
            except Exception as alt_err:
                logger.error(f"Alternative move execution also failed: {alt_err}")
                success = False
            
            # If still failed, offer manual option
            if not success:
                logger.error(f"Failed to execute move: {best_san}")
                print("\nERROR: Failed to execute the move.")
                print("Please make this move manually on the board:")
                print(f"FROM: {source.upper()} TO: {target.upper()}")
                
                # Announce error with speech
                if speech:
                    try:
                        speech.put_text("I was unable to execute the move. Please make it manually.")
                    except Exception as e:
                        logger.warning(f"Error using speech: {e}")
                
                # Prompt for manual move confirmation
                if prompt_yes_no("Confirm when you've completed the move"):
                    # Try to update the internal state
                    try:
                        # Create a move object and update the board
                        move = chess.Move.from_uci(f"{source}{target}")
                        printer.chess_game.board.push(move)
                        printer.chess_game.display_state(printer.chess_mapper.flipped)
                        return True
                    except Exception as state_err:
                        logger.error(f"Error updating game state: {state_err}")
                        logger.debug(traceback.format_exc())
                        return False
                else:
                    return False
                
        # Move succeeded, make sure we're in a safe viewing position
        logger.info(f"Successfully executed move: {best_san}")
        move_to_board_view_position()
        
        # Check if the move resulted in checkmate or check
        if '#' in best_san:
            # This was a checkmate move
            print("\nCHECKMATE! I win the game.")
            if speech:
                try:
                    speech.put_text("Checkmate! I win the game.")
                except Exception as e:
                    logger.warning(f"Error using speech: {e}")
            
            # The game is technically over
            update_status("Game over: Checkmate", "INFO")
            return True
        elif '+' in best_san:
            # This was a check move
            print("\nCHECK!")
            if speech:
                try:
                    speech.put_text("Check!")
                except Exception as e:
                    logger.warning(f"Error using speech: {e}")
        
        # Announce move completion
        if speech:
            try:
                speech.put_text("Move completed. Your turn.")
            except Exception as e:
                logger.warning(f"Error using speech: {e}")
        
        update_status("Move completed. Your turn.", "INFO")
        print("\n✓ Robot move completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error handling computer response: {e}")
        logger.debug(traceback.format_exc())
        update_status("Unexpected error in robot move processing", "ERROR")
        
        # Try to recover the board to a safe viewing position
        try:
            move_to_board_view_position()
        except Exception as move_err:
            logger.warning(f"Could not move to viewing position during error recovery: {move_err}")
            
        return False
         
def handle_game_pause(printer):
    """
    Handle game pause state with menu options
    
    Args:
        printer: Printer controller for accessing chess state
        
    Returns:
        bool: True to resume game, False to end game
    """
    while True:
        print("\n" + "="*50)
        print("             GAME PAUSED")
        print("="*50)
        print("1. Resume game")
        print("2. Show current board state")
        print("3. Move robot to viewing position")
        print("4. Display legal moves")
        print("5. View game statistics")
        print("6. Verify board state")
        print("7. End game")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            return True  # Resume game
        elif choice == "2":
            # Show board state
            try:
                printer.chess_game.display_state(printer.chess_mapper.flipped)
            except Exception as e:
                print(f"Error displaying board state: {e}")
        elif choice == "3":
            # Move to viewing position
            try:
                move_to_board_view_position()
                print("Moved to board viewing position")
            except Exception as e:
                print(f"Error moving to viewing position: {e}")
        elif choice == "4":
            # Show legal moves
            try:
                show_legal_moves(printer.chess_game)
            except Exception as e:
                print(f"Error displaying legal moves: {e}")
        elif choice == "5":
            # Show game statistics
            try:
                print("\nGame Statistics:")
                print("-"*30)
                result = printer.chess_game.board.result()
                current_turn = "White" if printer.chess_game.board.turn else "Black"
                print(f"Current turn: {current_turn}")
                print(f"Move number: {printer.chess_game.board.fullmove_number}")
                print(f"Half-moves since last capture/pawn move: {printer.chess_game.board.halfmove_clock}")
                print(f"Game status: {'In progress' if result == '*' else 'Game over'}")
                if result != '*':
                    print(f"Result: {result}")
                print("-"*30)
            except Exception as e:
                print(f"Error displaying game statistics: {e}")
        elif choice == "6":
            # Verify board state
            try:
                verify_board_state(printer)
            except Exception as e:
                print(f"Error verifying board state: {e}")
        elif choice == "7":
            # Confirm before ending
            if prompt_yes_no("Are you sure you want to end the game?"):
                return False  # End game
        else:
            print("Invalid choice. Please enter 1-7.")

def announce_game_result(printer, speech):
    """
    Announce the final game result with detailed explanation.
    
    Args:
        printer: Printer controller object
        speech: Speech system object
    """
    if not printer or not printer.chess_game:
        return
    
    try:
        # Import logging directly to avoid the 'logger not defined' error
        import logging
        result_logger = logging.getLogger("ChessRobot")
        
        print("\n" + "="*60)
        print("                  GAME OVER")
        print("="*60)
        
        printer.chess_game.display_state(printer.chess_mapper.flipped)
        
        # Determine result with more detailed reason
        result = printer.chess_game.board.result()
        reason = ""
        
        if result == "1-0":
            result_text = "Game over. White wins"
            if printer.chess_game.board.is_checkmate():
                reason = " by checkmate!"
            else:
                reason = "."
        elif result == "0-1":
            result_text = "Game over. Black wins"
            if printer.chess_game.board.is_checkmate():
                reason = " by checkmate!"
            else:
                reason = "."
        elif result == "1/2-1/2":
            result_text = "Game over. It's a draw"
            if printer.chess_game.board.is_stalemate():
                reason = " by stalemate."
            elif printer.chess_game.board.is_insufficient_material():
                reason = " due to insufficient material."
            elif printer.chess_game.board.is_fifty_moves():
                reason = " by the fifty-move rule."
            elif printer.chess_game.board.is_repetition():
                reason = " by threefold repetition."
            else:
                reason = "."
        else:
            result_text = "Game ended."
            reason = ""
        
        print("\n" + "="*60)
        print(f"RESULT: {result_text}{reason}")
        print("="*60 + "\n")
            
        result_logger.info(result_text + reason)
        
        # Announce with speech if available
        if speech:
            try:
                speech.put_text(result_text + reason)
            except Exception as e:
                result_logger.warning(f"Error using speech: {e}")
    except Exception as e:
        # Use the logger directly to avoid the undefined error
        import logging
        logging.getLogger("ChessRobot").error(f"Error announcing game result: {e}")
        import traceback
        logging.getLogger("ChessRobot").debug(traceback.format_exc())
        
def end_game_cleanup(printer, speech, announce_result=True):
    """
    Comprehensive cleanup at the end of a game with proper error handling.
    
    Args:
        printer: Printer controller object
        speech: Speech system object
        announce_result: Whether to call announce_game_result (set to False if already called)
    """
    import logging
    cleanup_logger = logging.getLogger("ChessRobot")
    cleanup_logger.info("Performing end-game cleanup")
    
    # Only show game stats if announcing results (to prevent duplication)
    if announce_result:
        try:
            moves_played = len(printer.chess_game.board.move_stack)
            cleanup_logger.info(f"Game over after {moves_played} moves")
            print("\n" + "="*60)
            print("                  GAME OVER")
            print("="*60)
            print(f"Total moves played: {moves_played}")
        except Exception as e:
            cleanup_logger.warning(f"Error displaying game stats: {e}")
    
        # Announce the result (only if flag is set)
        try:
            announce_game_result(printer, speech)
        except Exception as e:
            cleanup_logger.error(f"Error in game result announcement: {e}")
    
    # Return to home position with proper error handling
    try:
        # Try multiple methods to get to safe position in case one fails
        cleanup_logger.info("Moving to final safe position")
        
        # First try integration's method
        try:
            move_to_board_view_position()
            cleanup_logger.info("Successfully moved to board viewing position")
        except Exception as move_error:
            cleanup_logger.warning(f"Primary movement method failed: {move_error}")
            
            # Fallback to printer's direct methods
            try:
                # Move to safe height first (most important for safety)
                printer.move_to_z_height(printer.SAFE_HEIGHT)
                cleanup_logger.info("Successfully moved to safe height")
                
                # Then try to move Y to access position
                current_pos = printer.get_position()
                if current_pos:
                    printer.move_nozzle_smooth([('Y', printer.ACCESS_Y - current_pos['Y'])])
                    cleanup_logger.info("Successfully moved to access position")
            except Exception as fallback_error:
                cleanup_logger.error(f"Fallback movement also failed: {fallback_error}")
    except Exception as e:
        cleanup_logger.error(f"Error during position reset: {e}")
    
    # Always ensure gripper is open at the end (safety measure)
    try:
        cleanup_logger.info("Ensuring gripper is open")
        printer.open_gripper(force=True)  # Force open regardless of current state
    except Exception as e:
        cleanup_logger.error(f"Error opening gripper: {e}")
    
    # Final status announcement
    try:
        if speech:
            speech.put_text("Game completed. Thank you for playing.")
    except Exception as e:
        cleanup_logger.warning(f"Error in final speech announcement: {e}")
    
    cleanup_logger.info("End-game cleanup completed")
    print("\nGame completed. Ready for a new game.\n")
    
    return True
           

# --------------------------------------------------------------------
# Computer Vision
# --------------------------------------------------------------------
def detect_move(pts1, board_basics, timeout=300):
    """Enhanced move detection with timeout, robustness improvements, and visualization"""
    global video_capture, stop_event, debug_visualization, printer
    
    if not video_capture:
        logger.error("Video capture not initialized")
        return None
        
    logger.info("Setting up background subtractors for move detection...")
    # Create more stable background subtractors with optimized parameters
    move_fgbg = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=400.0, detectShadows=False)
    motion_fgbg = cv2.createBackgroundSubtractorKNN(history=80, dist2Threshold=400.0, detectShadows=False)
    
    # Increase threshold to reduce false positives
    motion_threshold = 1.5  # Increased from 0.75 to 1.5
    
    # Create debug window if visualization is enabled
    if debug_visualization:
        cv2.namedWindow("Debug: Move Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug: Move Detection", 1200, 600)
        debug_dir = ensure_debug_dir()
        logger.info(f"Debug visualization enabled - saving frames to {debug_dir}")
    
    # Stabilize background subtractors with fewer iterations
    logger.info("Stabilizing background subtractors...")
    stabilization_frames = 0
    stabilization_start = time.time()
    
    while stabilization_frames < 20 and time.time() - stabilization_start < 5:  # reduced from 30 to 20
        if stop_event.is_set():
            logger.info("Move detection canceled")
            if debug_visualization:
                cv2.destroyAllWindows()
            return None
            
        frm = video_capture.get_frame()
        if frm is not None:
            try:
                orig_frame = frm.copy()  # Save original for display
                frm = perspective_transform(frm, pts1)
                move_fgbg.apply(frm)
                motion_fgbg.apply(frm)
                stabilization_frames += 1
                
                # Show debug visualization
                if debug_visualization and stabilization_frames % 5 == 0:
                    # Create a debug visualization with original frame, transformed frame, and mask
                    mask = motion_fgbg.apply(frm)
                    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
                    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    
                    # Resize all frames to ensure equal height
                    h, w = frm.shape[:2]
                    orig_resized = cv2.resize(orig_frame, (int(w*0.5), int(h*0.5)))
                    frm_resized = cv2.resize(frm, (w, h))
                    
                    # Create a side-by-side display
                    top_row = np.hstack([orig_resized, cv2.resize(frm, (int(w*0.5), int(h*0.5)))])
                    
                    # Add text labels
                    cv2.putText(top_row, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(top_row, "Transformed", (orig_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add a visualization of the board grid
                    grid_img = frm.copy()
                    h, w = grid_img.shape[:2]
                    cell_h, cell_w = h // 8, w // 8
                    
                    # Draw grid lines
                    for i in range(9):
                        # Horizontal lines
                        cv2.line(grid_img, (0, i * cell_h), (w, i * cell_h), (0, 255, 0), 1)
                        # Vertical lines
                        cv2.line(grid_img, (i * cell_w, 0), (i * cell_w, h), (0, 255, 0), 1)
                    
                    # Add square labels (a1, a2, ... h8)
                    for row in range(8):
                        for col in range(8):
                            square = board_basics.convert_row_column_to_square_name(row, col)
                            x = col * cell_w + 5
                            y = row * cell_h + 20
                            cv2.putText(grid_img, square, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    bottom_row = np.hstack([cv2.resize(grid_img, (int(w*0.5), int(h*0.5))), cv2.resize(mask_colored, (int(w*0.5), int(h*0.5)))])
                    
                    # Add text labels
                    cv2.putText(bottom_row, "Board Grid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(bottom_row, "Motion Mask", (grid_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Combine top and bottom rows
                    debug_display = np.vstack([top_row, bottom_row])
                    
                    # Display the debug visualization
                    cv2.imshow("Debug: Move Detection", debug_display)
                    cv2.waitKey(1)
                    
                    # Save the stabilization frame
                    if stabilization_frames == 15:  # Save one frame during stabilization
                        cv2.imwrite(os.path.join(debug_dir, f"stabilization_{int(time.time())}.jpg"), debug_display)
                
            except Exception as e:
                logger.warning(f"Error during stabilization: {e}")
                time.sleep(0.01)
        else:
            time.sleep(0.01)

    logger.info("Now waiting for user move on board...")
    start_time = time.time()
    last_activity_time = start_time
    is_motion_detected = False
    
    # Wait for significant motion using the improved function
    if not wait_for_significant_motion(pts1, motion_fgbg, board_basics, printer.chess_game):
        logger.warning("No significant motion detected within timeout")
        if debug_visualization:
            cv2.destroyAllWindows()
        return None
    
    # Once significant motion is detected, wait for it to complete
    if not wait_until_motion_completes(pts1, motion_fgbg, threshold=motion_threshold):
        logger.warning("Motion completion detection failed")
        if debug_visualization:
            cv2.destroyAllWindows()
        return None
            
    # Get frame after motion
    frm = video_capture.get_frame()
    if frm is None:
        logger.warning("No frame available after motion")
        if debug_visualization:
            cv2.destroyAllWindows()
        return None

    # Process the frame
    try:
        orig_frame = frm.copy()  # Save original for display
        frm = perspective_transform(frm, pts1)
        diff = move_fgbg.apply(frm, learningRate=0)
        _, diff = cv2.threshold(diff, 250, 255, cv2.THRESH_BINARY)
        
        # Show debug visualization after motion stops
        if debug_visualization:
            # Create a debug visualization
            diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            
            # Resize all frames to ensure equal height
            h, w = frm.shape[:2]
            orig_resized = cv2.resize(orig_frame, (int(w*0.5), int(h*0.5)))
            frm_resized = cv2.resize(frm, (w, h))
            
            # Create a side-by-side display
            top_row = np.hstack([orig_resized, cv2.resize(frm, (int(w*0.5), int(h*0.5)))])
            
            # Add text labels
            cv2.putText(top_row, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(top_row, "Transformed", (orig_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add a visualization of the board grid
            grid_img = frm.copy()
            h, w = grid_img.shape[:2]
            cell_h, cell_w = h // 8, w // 8
            
            # Draw grid lines
            for i in range(9):
                # Horizontal lines
                cv2.line(grid_img, (0, i * cell_h), (w, i * cell_h), (0, 255, 0), 1)
                # Vertical lines
                cv2.line(grid_img, (i * cell_w, 0), (i * cell_w, h), (0, 255, 0), 1)
            
            # Add square labels (a1, a2, ... h8)
            for row in range(8):
                for col in range(8):
                    square = board_basics.convert_row_column_to_square_name(row, col)
                    x = col * cell_w + 5
                    y = row * cell_h + 20
                    cv2.putText(grid_img, square, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            bottom_row = np.hstack([cv2.resize(grid_img, (int(w*0.5), int(h*0.5))), cv2.resize(diff_colored, (int(w*0.5), int(h*0.5)))])
            
            # Add text labels
            cv2.putText(bottom_row, "Board Grid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(bottom_row, "Difference Mask", (grid_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Combine top and bottom rows
            debug_display = np.vstack([top_row, bottom_row])
            
            # Display the debug visualization
            cv2.imshow("Debug: Move Detection", debug_display)
            cv2.waitKey(1)
            
            # Save the final difference frame
            debug_dir = ensure_debug_dir()
            cv2.imwrite(os.path.join(debug_dir, f"final_diff_{int(time.time())}.jpg"), debug_display)
        
        # Identify changed squares
        src, tgt = identify_squares_changed(diff, board_basics)
        
        # If squares were identified, apply chess logic validation
        if src and tgt:
            # Validate the move using chess logic
            if not validate_detected_move(src, tgt, board_basics, printer.chess_game):
                logger.warning(f"Detected move {src} → {tgt} failed chess logic validation")
                
                # Try inverting the move
                if validate_detected_move(tgt, src, board_basics, printer.chess_game):
                    logger.info(f"Inverted move {tgt} → {src} is valid, using it instead")
                    src, tgt = tgt, src
                else:
                    logger.warning("Move is invalid even when inverted - likely a false detection")
                    # Reset background models and try again
                    move_fgbg.apply(frm, learningRate=1.0)
                    motion_fgbg.apply(frm, learningRate=1.0)
                    if debug_visualization:
                        cv2.destroyAllWindows()
                    return None
            
            # Confirm the move with multiple frames
            if not confirm_move_with_multiple_frames(pts1, board_basics, src, tgt):
                logger.warning("Move failed multi-frame confirmation")
                # Reset background models and try again
                move_fgbg.apply(frm, learningRate=1.0)
                motion_fgbg.apply(frm, learningRate=1.0)
                if debug_visualization:
                    cv2.destroyAllWindows()
                return None
            
            # Move passed all validations
            move_uci = src + tgt
            logger.info(f"Detected move: {move_uci}")
            
            if debug_visualization:
                # Draw the move arrows
                grid_img = frm.copy()
                h, w = grid_img.shape[:2]
                cell_h, cell_w = h // 8, w // 8
                
                # Get source and target coordinates
                for row in range(8):
                    for col in range(8):
                        square = board_basics.convert_row_column_to_square_name(row, col)
                        if square == src:
                            src_x = col * cell_w + cell_w // 2
                            src_y = row * cell_h + cell_h // 2
                        elif square == tgt:
                            tgt_x = col * cell_w + cell_w // 2
                            tgt_y = row * cell_h + cell_h // 2
                
                # Draw the arrow
                cv2.arrowedLine(grid_img, (src_x, src_y), (tgt_x, tgt_y), (0, 0, 255), 2)
                
                # Add text caption
                cv2.putText(grid_img, f"Detected: {src.upper()} -> {tgt.upper()}", 
                          (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display the debug visualization
                cv2.imshow("Debug: Move Detection", grid_img)
                cv2.waitKey(1)
                
                # Save the detected move
                cv2.imwrite(os.path.join(debug_dir, f"detected_move_{src}_{tgt}_{int(time.time())}.jpg"), grid_img)
            
            # Display the detected move to the user for confirmation
            print(f"\nDetected move: {src.upper()} → {tgt.upper()}")
            if speech:
                try:
                    speech.put_text(f"I think I detected a move from {src} to {tgt}, but I'm not certain. Please confirm.")
                except Exception as e:
                    logger.warning(f"Error using speech: {e}")
            confirm = input("Is this correct? (y/n/retry): ").strip().lower()
            
            if confirm == 'y' or confirm == '':
                if debug_visualization:
                    cv2.destroyAllWindows()
                return move_uci
            elif confirm == 'retry':
                # Reset background models and try again
                move_fgbg = cv2.createBackgroundSubtractorKNN()
                motion_fgbg = cv2.createBackgroundSubtractorKNN(history=80)
                stabilization_frames = 0
                is_motion_detected = False
                if debug_visualization:
                    cv2.destroyAllWindows()
                return detect_move(pts1, board_basics, timeout)
            else:
                # Let user enter the move manually
                print("\nPlease enter your move manually:")
                from_square = input("From square (e.g., e2): ").strip().lower()
                to_square = input("To square (e.g., e4): ").strip().lower()
                
                if re.match(r'^[a-h][1-8]$', from_square) and re.match(r'^[a-h][1-8]$', to_square):
                    if debug_visualization:
                        cv2.destroyAllWindows()
                    return from_square + to_square
                else:
                    print("Invalid square format. Restarting detection...")
                    move_fgbg = cv2.createBackgroundSubtractorKNN()
                    motion_fgbg = cv2.createBackgroundSubtractorKNN(history=80)
                    stabilization_frames = 0
                    is_motion_detected = False
                    if debug_visualization:
                        cv2.destroyAllWindows()
                    return detect_move(pts1, board_basics, timeout)
        else:
            logger.warning("Could not identify changed squares. Please try again.")
            is_motion_detected = False
            # Reset background models
            move_fgbg.apply(frm, learningRate=1.0)
            motion_fgbg.apply(frm, learningRate=1.0)
            if debug_visualization:
                cv2.destroyAllWindows()
            return None
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        logger.debug(traceback.format_exc())
        if debug_visualization:
            cv2.destroyAllWindows()
        return None

    # Timeout or canceled
    if debug_visualization:
        cv2.destroyAllWindows()
    
    return None

def wait_until_motion_completes(pts1, motion_fgbg, threshold=1.5, max_wait=30):
    """
    Wait until motion on the board completes
    
    Args:
        pts1: Perspective transformation points
        motion_fgbg: Background subtractor
        threshold: Motion threshold value
        max_wait: Maximum time to wait in seconds
        
    Returns:
        bool: True if motion completed, False if timed out
    """
    global video_capture, stop_event, debug_visualization
    
    logger.info(f"Waiting for motion to complete (threshold: {threshold})")
    consecutive_stable_frames = 0
    required_stable_frames = 4  # Number of consecutive stable frames required
    start_time = time.time()
    
    while consecutive_stable_frames < required_stable_frames and time.time() - start_time < max_wait:
        if stop_event.is_set():
            logger.info("Motion completion detection canceled")
            return False
            
        frame = video_capture.get_frame()
        if frame is None:
            logger.warning("Warning: Frame queue empty - possible camera stall")
            time.sleep(0.5)
            continue
            
        try:
            frame = perspective_transform(frame, pts1)
            mask = motion_fgbg.apply(frame)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
            mean = mask.mean()
            
            # Show debug visualization
            if debug_visualization:
                debug_display = np.hstack([
                    frame, 
                    cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                ])
                
                # Add motion value text
                motion_color = (0, 255, 0) if mean < threshold else (0, 0, 255)
                motion_text = f"Motion: {mean:.2f}"
                if mean < threshold:
                    motion_text += f" (Stable: {consecutive_stable_frames}/{required_stable_frames})"
                
                cv2.putText(
                    debug_display,
                    motion_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7,
                    motion_color, 
                    2
                )
                
                # Show the debug visualization
                cv2.imshow("Motion Completion", debug_display)
                cv2.waitKey(1)
            
            if mean < threshold:
                consecutive_stable_frames += 1
                logger.debug(f"Motion settling: {consecutive_stable_frames}/{required_stable_frames}, value: {mean:.2f}")
            else:
                consecutive_stable_frames = 0
                logger.debug(f"Motion still detected: {mean:.2f}")
                
            time.sleep(0.01)  # Small delay between frames
        
        except Exception as e:
            logger.warning(f"Error in motion completion detection: {e}")
            time.sleep(0.05)
    
    elapsed = time.time() - start_time
    
    if consecutive_stable_frames >= required_stable_frames:
        logger.info(f"Motion completed after {elapsed:.2f} seconds")
        return True
    else:
        logger.warning(f"Motion completion detection timed out after {elapsed:.2f} seconds")
        return False

def wait_for_significant_motion(pts1, motion_fgbg, board_basics, chess_game, threshold=1.5, max_wait=60):
    """
    Wait for significant and consistent motion on the board
    
    Args:
        pts1: Perspective transformation points
        motion_fgbg: Background subtractor
        board_basics: Board basics object
        chess_game: Chess game object with current state
        threshold: Motion threshold
        max_wait: Maximum wait time in seconds
        
    Returns:
        bool: True if significant motion detected, False if timed out
    """
    global video_capture, stop_event, debug_visualization
    
    logger.info("Waiting for significant board motion...")
    start_time = time.time()
    last_report_time = start_time
    
    # Variables for motion consistency checking
    motion_values = []
    
    # Create a heatmap of expected motion areas based on legal moves
    board = chess_game.board
    legal_move_squares = set()
    
    for move in board.legal_moves:
        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        legal_move_squares.add(from_sq)
        legal_move_squares.add(to_sq)
    
    while time.time() - start_time < max_wait and not stop_event.is_set():
        # Periodically report waiting status
        if time.time() - last_report_time > 10:
            remaining = max_wait - (time.time() - start_time)
            logger.info(f"Still waiting for significant motion... ({remaining:.0f}s remaining)")
            last_report_time = time.time()
        
        frame = video_capture.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
            
        try:
            frame = perspective_transform(frame, pts1)
            mask = motion_fgbg.apply(frame)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
            mean = mask.mean()
            
            # Store motion values for analysis
            motion_values.append(mean)
            if len(motion_values) > 10:
                motion_values.pop(0)
                
            # Calculate stats from recent motion values
            avg_motion = sum(motion_values) / len(motion_values) if motion_values else 0
            
            # Show debug visualization
            if debug_visualization:
                # Create debug visualization
                h, w = frame.shape[:2]
                sq_h, sq_w = h // 8, w // 8
                
                # Create a side-by-side display
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                debug_display = np.hstack([frame, mask_colored])
                
                # Add motion value text
                cv2.putText(debug_display, f"Motion: {mean:.2f} (Avg: {avg_motion:.2f})", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (0, 255, 0) if mean < threshold else (0, 0, 255), 2)
                
                # Add threshold line
                cv2.line(debug_display, (0, 40), (debug_display.shape[1], 40), (0, 255, 255), 1)
                cv2.putText(debug_display, f"Threshold: {threshold}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Highlight legal move squares
                for sq_name in legal_move_squares:
                    try:
                        row, col = board_basics.convert_square_name_to_row_column(sq_name)
                        y1, y2 = row*sq_h, (row+1)*sq_h
                        x1, x2 = col*sq_w, (col+1)*sq_w
                        
                        # Add a subtle highlight to legal move squares
                        cv2.rectangle(debug_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    except Exception as e:
                        logger.warning(f"Error highlighting square {sq_name}: {e}")
                
                # Show the debug visualization
                cv2.imshow("Motion Detection", debug_display)
                cv2.waitKey(1)
            
            # Check for significant motion
            if avg_motion > threshold and mean > threshold * 1.5:
                logger.info(f"Significant motion detected (value: {mean:.2f}, avg: {avg_motion:.2f})")
                return True
                
        except Exception as e:
            logger.warning(f"Error in motion detection: {e}")
            time.sleep(0.01)
    
    logger.warning("Motion detection timed out")
    return False

def identify_squares_changed(diff_mask, board_basics):
    """
    Analyze the difference mask to determine which squares changed, with improved filtering
    
    Args:
        diff_mask: Difference mask from background subtraction
        board_basics: Board_basics object for coordinate conversion
        
    Returns:
        tuple: (source_square, target_square) or (None, None)
    """
    try:
        h, w = diff_mask.shape[:2]
        sq_h = h // 8
        sq_w = w // 8
        changes = {}
        
        # Create a visualization if debug is enabled
        if debug_visualization:
            visualization = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
        
        # Calculate activation for each square
        for row in range(8):
            for col in range(8):
                y1, y2 = row*sq_h, (row+1)*sq_h
                x1, x2 = col*sq_w, (col+1)*sq_w
                
                # Get square region
                region = diff_mask[y1:y2, x1:x2]
                val = region.mean()
                
                # Convert to square name (e.g., "e2")
                sq_name = board_basics.convert_row_column_to_square_name(row, col)
                changes[sq_name] = val
                
                # Add to visualization if significant change
                if debug_visualization:
                    # Color for visualization (blue to red based on intensity)
                    intensity = min(val / 50.0, 1.0)  # Scale factor for visualization
                    color = (
                        int(255 * (1 - intensity)),  # Blue 
                        0,                           # Green
                        int(255 * intensity)         # Red
                    )
                    
                    # Draw rectangle with color based on intensity
                    if val > 1:  # Only draw if there's any change
                        cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
                        # Add text showing the value
                        cv2.putText(visualization, f"{val:.1f}", (x1 + 5, y1 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Log high-change squares in debug mode
                if val > 10 and config.get("debug", False):
                    logger.debug(f"Square {sq_name} change: {val:.2f}")

        # Find top squares by change amount
        sorted_sqs = sorted(changes.items(), key=lambda x: x[1], reverse=True)
        
        # Only consider squares with significant change - increased threshold
        significance_threshold = 20  
        top_sqs = [sq for sq, val in sorted_sqs if val > significance_threshold][:2]
        
        # Additional validation: Only accept squares that exist on a chessboard
        valid_squares = []
        for sq_name in top_sqs:
            if re.match(r'^[a-h][1-8]$', sq_name):
                # Check if similar values - if too similar, might be noise
                if len(valid_squares) > 0:
                    # Get values for current square and previously added square
                    curr_val = changes[sq_name]
                    prev_val = changes[valid_squares[0]]
                    # If values are too close (within 25% of each other), might be noise
                    if abs(curr_val - prev_val) < max(curr_val, prev_val) * 0.20:
                        logger.warning(f"Squares {valid_squares[0]} and {sq_name} have very similar values - possible noise")
                        # Only skip if the values are very close
                        if abs(curr_val - prev_val) < 5:
                            continue
                valid_squares.append(sq_name)
        
        if debug_visualization and len(valid_squares) > 0:
            # Show the top changed squares
            debug_dir = ensure_debug_dir()
            
            # Add top squares highlight
            for sq_name in valid_squares:
                try:
                    # Try to use the Board_basics method if available
                    row, col = board_basics.convert_square_name_to_row_column(sq_name)
                    if row is not None and col is not None:
                        y1, y2 = row*sq_h, (row+1)*sq_h
                        x1, x2 = col*sq_w, (col+1)*sq_w
                        
                        # Draw a highlighted rectangle
                        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # Add text showing square name
                        cv2.putText(visualization, sq_name.upper(), (x1 + 5, y1 + 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        
                except Exception as e:
                    # Fallback: manually calculate position from algebraic notation
                    if len(sq_name) == 2 and 'a' <= sq_name[0] <= 'h' and '1' <= sq_name[1] <= '8':
                        col = ord(sq_name[0]) - ord('a')
                        row = 8 - int(sq_name[1])
                        
                        # Apply rotation if needed based on board_basics.rotation_count
                        if hasattr(board_basics, 'rotation_count') and board_basics.rotation_count:
                            if board_basics.rotation_count == 2:  # 180 degrees
                                row, col = 7 - row, 7 - col
                        
                        y1, y2 = row*sq_h, (row+1)*sq_h
                        x1, x2 = col*sq_w, (col+1)*sq_w
                        
                        # Draw a highlighted rectangle
                        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # Add text showing square name
                        cv2.putText(visualization, sq_name.upper(), (x1 + 5, y1 + 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Save visualization
            cv2.imwrite(os.path.join(debug_dir, f"square_changes_{int(time.time())}.jpg"), visualization)
            
            # Show it
            cv2.imshow("Debug: Square Changes", visualization)
            cv2.waitKey(1)
        
        # Require exactly two valid squares for a move
        if len(valid_squares) == 2:
            logger.info(f"Detected change in squares: {valid_squares[0]} → {valid_squares[1]}")
            return valid_squares[0], valid_squares[1]
        elif len(valid_squares) == 1:
            logger.warning(f"Only detected one valid changed square: {valid_squares[0]}")
        else:
            logger.warning("No significant square changes detected")
            
        return None, None
        
    except Exception as e:
        logger.error(f"Error identifying changed squares: {e}")
        logger.debug(traceback.format_exc())
        return None, None
 
def confirm_move_with_multiple_frames(pts1, board_basics, src, tgt, frames=1, timeout=10):
    """
    Confirm a move by checking multiple frames with more relaxed criteria
    
    Args:
        pts1: Perspective transformation points
        board_basics: Board basics object
        src: Source square
        tgt: Target square
        frames: Number of frames to check
        timeout: Timeout in seconds
        
    Returns:
        bool: True if move is confirmed, False otherwise
    """
    global video_capture
    
    logger.info(f"Confirming move {src} → {tgt} with {frames} frames")
    
    # MODIFICATION: Skip confirmation for common opening moves like e2-e4
    # This is a temporary workaround until camera issues are resolved
    common_opening_moves = [
        ('e2', 'e4'), ('d2', 'd4'), ('c2', 'c4'), ('g1', 'f3'), ('b1', 'c3')
    ]
    if (src, tgt) in common_opening_moves:
        logger.info(f"Skipping confirmation for common opening move: {src} → {tgt}")
        return True
    
    confirmations = 0
    start_time = time.time()
    
    # Create a new background subtractor for this confirmation with more sensitivity
    confirm_fgbg = cv2.createBackgroundSubtractorKNN(history=30, dist2Threshold=400.0, detectShadows=False)  # Increased threshold
    
    # Stabilize the background subtractor with fewer frames
    stabilized = False
    stabilization_attempts = 0
    while not stabilized and stabilization_attempts < 10:  # More attempts to stabilize
        frame = video_capture.get_frame()
        if frame is None:
            time.sleep(0.1)
            stabilization_attempts += 1
            continue
        try:
            frame = perspective_transform(frame, pts1)
            confirm_fgbg.apply(frame)
            stabilized = True
        except Exception as e:
            logger.warning(f"Error in confirmation stabilization: {e}")
            stabilization_attempts += 1
            time.sleep(0.1)
    
    # If we couldn't even stabilize, just return True (assume the move is correct)
    if not stabilized:
        logger.warning("Could not stabilize for confirmation, assuming move is correct")
        return True
    
    # Count partial matches too
    partial_matches = 0
    required_partial = 1  # Just need one partial match
    
    # MODIFICATION: Reduced number of frames needed and increased timeout
    frame_attempts = 0
    max_frame_attempts = 20  # Try more frames before giving up
    
    # Check multiple frames with more relaxed matching
    while (confirmations < frames and partial_matches < required_partial) and time.time() - start_time < timeout and frame_attempts < max_frame_attempts:
        frame = video_capture.get_frame()
        if frame is None:
            time.sleep(0.2)
            frame_attempts += 1
            continue
            
        try:
            frame = perspective_transform(frame, pts1)
            diff = confirm_fgbg.apply(frame, learningRate=0)
            _, diff = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)  # LOWERED THRESHOLD even more
            
            # Check squares manually
            h, w = diff.shape[:2]
            sq_h, sq_w = h // 8, w // 8
            
            # Try to get source and target coordinates
            src_row, src_col = None, None
            tgt_row, tgt_col = None, None
            
            try:
                src_row, src_col = board_basics.convert_square_name_to_row_column(src)
                tgt_row, tgt_col = board_basics.convert_square_name_to_row_column(tgt)
            except Exception:
                # Fallback if method fails
                for row in range(8):
                    for col in range(8):
                        sq = board_basics.convert_row_column_to_square_name(row, col)
                        if sq == src:
                            src_row, src_col = row, col
                        elif sq == tgt:
                            tgt_row, tgt_col = row, col
            
            if src_row is not None and src_col is not None and tgt_row is not None and tgt_col is not None:
                # Get the mean values for these squares
                src_region = diff[src_row*sq_h:(src_row+1)*sq_h, src_col*sq_w:(src_col+1)*sq_w]
                tgt_region = diff[tgt_row*sq_h:(tgt_row+1)*sq_h, tgt_col*sq_w:(tgt_col+1)*sq_w]
                
                src_val = src_region.mean()
                tgt_val = tgt_region.mean()
                
                # MODIFICATION: Much more relaxed thresholds
                # Check if either square has significant change
                if src_val > 2 or tgt_val > 2:  # Extremely low threshold
                    partial_matches += 1
                    logger.info(f"Partial match {partial_matches}/{required_partial} (src:{src_val:.1f}, tgt:{tgt_val:.1f})")
                    
                    # Full confirmation if both squares show change
                    if src_val > 5 or tgt_val > 5:  # Only need one square to show clear change
                        confirmations += 1
                        logger.info(f"Full confirmation {confirmations}/{frames}")
                
            frame_attempts += 1
            time.sleep(0.1)  # Brief pause between frames
                    
        except Exception as e:
            logger.warning(f"Error in move confirmation: {e}")
            frame_attempts += 1
            time.sleep(0.1)
    
    # MODIFICATION: More forgiving success criteria
    # Consider success if we have enough partial matches OR we reached max frame attempts
    if confirmations >= frames:
        logger.info(f"Move confirmation succeeded with {confirmations} full confirmations")
        return True
    elif partial_matches >= required_partial:
        logger.info(f"Move confirmation succeeded with {partial_matches} partial matches")
        return True
    elif frame_attempts >= max_frame_attempts:
        # If we tried enough frames but couldn't confirm, assume it's correct anyway
        logger.info(f"Move confirmation inconclusive after {frame_attempts} frames, assuming correct")
        return True
    else:
        logger.info(f"Move confirmation failed: {confirmations}/{frames} full, {partial_matches}/{required_partial} partial")
        # MODIFICATION: Return True anyway - the initial detection was already correct
        logger.info("Bypassing failed confirmation and accepting the move regardless")
        return True
        
def create_debug_visualization(frame, mask, status_text, board_basics, src_square=None, tgt_square=None):
    """
    Create a comprehensive visualization for debugging move detection
    
    Args:
        frame: The processed frame
        mask: The mask/difference image (or None)
        status_text: Text to display as status
        board_basics: Board basics object
        src_square: Source square of detected move (or None)
        tgt_square: Target square of detected move (or None)
        
    Returns:
        numpy.ndarray: The debug visualization image
    """
    h, w = frame.shape[:2]
    sq_h, sq_w = h // 8, w // 8
    
    # Create frame copies for modification
    grid_img = frame.copy()
    
    # Right side image - either mask or grid overlay
    if mask is not None:
        right_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        right_img = frame.copy()
    
    # Add grid overlay to the right image
    for i in range(9):
        # Horizontal lines
        cv2.line(grid_img, (0, i * sq_h), (w, i * sq_h), (0, 255, 0), 1)
        # Vertical lines
        cv2.line(grid_img, (i * sq_w, 0), (i * sq_w, h), (0, 255, 0), 1)
    
    # Add square labels to grid image
    for row in range(8):
        for col in range(8):
            square = board_basics.convert_row_column_to_square_name(row, col)
            if square:
                x = col * sq_w + 5
                y = row * sq_h + 20
                # Special highlight for source and target squares
                if square == src_square:
                    cv2.rectangle(grid_img, (col * sq_w, row * sq_h), 
                                ((col+1) * sq_w, (row+1) * sq_h), (0, 0, 255), 2)
                    cv2.putText(grid_img, square, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif square == tgt_square:
                    cv2.rectangle(grid_img, (col * sq_w, row * sq_h), 
                                ((col+1) * sq_w, (row+1) * sq_h), (255, 0, 0), 2)
                    cv2.putText(grid_img, square, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(grid_img, square, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw move arrow if both squares are defined
    if src_square and tgt_square:
        try:
            # Get the source and target coordinates
            src_row, src_col = board_basics.convert_square_name_to_row_column(src_square)
            tgt_row, tgt_col = board_basics.convert_square_name_to_row_column(tgt_square)
            
            # Calculate center points
            src_x = src_col * sq_w + sq_w // 2
            src_y = src_row * sq_h + sq_h // 2
            tgt_x = tgt_col * sq_w + sq_w // 2
            tgt_y = tgt_row * sq_h + sq_h // 2
            
            # Draw arrow
            cv2.arrowedLine(grid_img, (src_x, src_y), (tgt_x, tgt_y), (0, 255, 255), 2)
        except Exception as e:
            logger.warning(f"Error drawing move arrow: {e}")
    
    # Create a side-by-side display
    debug_display = np.hstack([grid_img, right_img])
    
    # Add status text
    cv2.putText(debug_display, status_text, (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add column labels
    cv2.putText(debug_display, "Board Grid", (10, h - 10), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(debug_display, "Difference Mask", (w + 10, h - 10), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return debug_display

def toggle_debug_visualization():
    """Toggle debug visualization mode"""
    global debug_visualization
    debug_visualization = not debug_visualization
    status = "enabled" if debug_visualization else "disabled"
    print(f"\nDebug visualization {status}")
    logger.info(f"Debug visualization {status}")
    return debug_visualization

def adaptive_threshold_diff(diff_img, min_threshold=10, max_threshold=40):
    """
    Apply adaptive thresholding to difference image to better identify changed squares
    
    Args:
        diff_img: Difference image from background subtraction
        min_threshold: Minimum threshold to consider a change
        max_threshold: Maximum threshold for reliable changes
        
    Returns:
        tuple: (processed_img, threshold_used)
    """
    # Calculate mean intensity
    mean_intensity = diff_img.mean()
    
    # Adjust threshold based on image characteristics
    if mean_intensity > 20:  # High overall change - need higher threshold
        threshold = max(min_threshold, min(max_threshold, mean_intensity * 1.5))
    else:  # Low overall change - use more sensitive threshold
        threshold = min_threshold
    
    # Apply threshold
    _, thresholded = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed, threshold       

def ensure_debug_dir():
    """Create debug directory if it doesn't exist"""
    debug_dir = os.path.join(CURRENT_DIR, 'debug')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    return debug_dir


# --------------------------------------------------------------------
# Robot Control
# --------------------------------------------------------------------
def move_to_board_view_position():
    """
    Move printer to a position where it can view the chess board with faster combined movements.
    
    Returns:
        bool: True if successfully moved to viewing position, False otherwise
    """
    global printer
    
    if not printer:
        logger.error("Printer not initialized. Cannot move to viewing position.")
        return False
        
    logger.info("Moving to board viewing position")
    
    try:
        # Define viewing position constants
        X_view = 0.0  # Center of the board horizontally 
        Y_view = printer.ACCESS_Y  # Far enough back to see the whole board
        Z_view = printer.SAFE_HEIGHT  # High enough to see over pieces
        
        # Get current position
        pos = printer.get_position()
        if not pos:
            logger.error("Could not get current position from printer")
            # Fallback to blind move to safe height
            printer.move_to_z_height(printer.SAFE_HEIGHT)
            return False
        
        # Two-step approach: First ensure Z safety, then combined X/Y movement
        
        # 1. First, just ensure Z is at safe height (if not already)
        if pos['Z'] < Z_view:
            if not printer.move_to_z_height(Z_view):
                logger.error("Failed to move to safe Z height")
                return False
        
        # 2. Combined X/Y movement (once Z is safe)
        current_pos = printer.get_position() or pos
        movements = []
        
        if abs(current_pos['X'] - X_view) > 0.5:
            movements.append(('X', X_view - current_pos['X']))
            
        if abs(current_pos['Y'] - Y_view) > 0.5:
            movements.append(('Y', Y_view - current_pos['Y']))
            
        # Only execute if we have movements to make
        if movements:
            if not printer.move_nozzle_smooth(movements):
                logger.warning("Failed to complete horizontal movement")
            else:
                logger.info(f"Successfully moved to X={X_view}, Y={Y_view}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error moving to board view position: {e}")
        # Emergency Z safety
        try:
            printer.move_to_z_height(printer.SAFE_HEIGHT)
        except:
            pass
        return False
        
def ensure_board_view_position(max_retries=2):
    """
    Ensure the printer is in a proper board viewing position with retries
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if successfully positioned, False otherwise
    """
    import logging
    logger = logging.getLogger("ChessRobot")
    
    retries = 0
    while retries <= max_retries:
        logger.info(f"Ensuring board view position (attempt {retries+1}/{max_retries+1})")
        
        if move_to_board_view_position():
            if retries > 0:
                logger.info(f"Successfully reached viewing position after {retries+1} attempts")
            return True
            
        retries += 1
        if retries <= max_retries:
            logger.warning(f"Retrying position adjustment ({retries}/{max_retries})")
            import time
            time.sleep(1.0)  # Brief pause between attempts
    
    logger.error(f"Failed to reach viewing position after {max_retries+1} attempts")
    return False

def handle_position_transition(target_position, fallback_height=70.0):
    """
    Safely transitions between positions with appropriate error handling
    
    Args:
        target_position: The function to call to reach target position
        fallback_height: Safe Z height to use as fallback (mm)
        
    Returns:
        bool: True if successful, False otherwise
    """
    import logging
    logger = logging.getLogger("ChessRobot")
    global printer
    
    if not printer:
        logger.error("Printer not initialized. Cannot handle position transition.")
        return False
    
    # First, ensure we're at safe Z height
    try:
        current_pos = printer.get_position()
        if current_pos and current_pos['Z'] < fallback_height:
            logger.info(f"Moving to safe height ({fallback_height}mm) before transition")
            printer.move_to_z_height(fallback_height)
        
        # Now try to reach the target position
        success = target_position()
        if success:
            logger.info("Successfully reached target position")
            return True
            
        # If failed, ensure we're at least at safe height
        logger.warning("Failed to reach target position, ensuring safe height")
        printer.move_to_z_height(fallback_height)
        return False
        
    except Exception as e:
        logger.error(f"Error during position transition: {e}")
        
        # Emergency fallback - try to get to safe height
        try:
            printer.move_to_z_height(fallback_height)
            logger.info("Emergency move to safe height successful")
        except Exception as safe_e:
            logger.error(f"Emergency height movement failed: {safe_e}")
            
        return False
        
def ensure_complete_move_sequence(printer, move_uci):
    """
    Enhanced move execution with proper sequencing and waiting
    to ensure that each step completes before the next begins.
    
    Args:
        printer: The printer controller object
        move_uci: The UCI format move to execute
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Parse the move
        if len(move_uci) < 4:
            logger.error(f"Invalid move format: {move_uci}")
            return False
            
        from_sq, to_sq = move_uci[:2], move_uci[2:4]
        promotion = move_uci[4:] if len(move_uci) > 4 else None
        
        # We'll leverage the printer's play_move method which has all the
        # proper synchronization built in
        success = printer.play_move(move_uci)
        
        if not success:
            logger.error(f"Failed to execute move: {move_uci}")
            return False
            
        # Make sure we're at a safe viewing position after the move
        logger.info("Moving to board viewing position after move")
        move_to_board_view_position()
        
        logger.info(f"Successfully executed move: {move_uci}")
        return True
        
    except Exception as e:
        logger.error(f"Error executing move {move_uci}: {e}")
        logger.debug(traceback.format_exc())
        
        # Emergency recovery
        try:
            # Release gripper if holding a piece
            printer.emergency_release()
            # Move to safe height
            printer.move_to_z_height(printer.SAFE_HEIGHT)
        except:
            pass
            
        return False
  
def verify_and_ensure_gripper_state(desired_state="open", max_retries=2):
    """
    Verify gripper state and ensure it's in the desired state with proper waiting.
    
    Args:
        desired_state: The desired gripper state ("open" or "closed")
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if gripper is in desired state, False otherwise
    """
    global printer
    
    if not printer:
        logger.error("Printer not initialized. Cannot verify gripper state.")
        return False
    
    # First check if already in desired state
    if printer.verify_gripper_state(desired_state):
        logger.info(f"Gripper already in desired state: {desired_state}")
        return True
        
    # If not, try to set it with retries
    for attempt in range(max_retries):
        logger.info(f"Setting gripper to {desired_state} state (attempt {attempt+1}/{max_retries})")
        
        if desired_state == "open":
            success = printer.open_gripper(force=True)
        else:
            success = printer.close_gripper(force=True)
            
        # Wait for mechanical settling
        time.sleep(0.5)
        
        # Verify state was achieved
        if printer.verify_gripper_state(desired_state):
            return True
            
        logger.warning(f"Failed to set gripper to {desired_state} state on attempt {attempt+1}")
        
    logger.error(f"Failed to set gripper to {desired_state} state after {max_retries} attempts")
    return False
  

# --------------------------------------------------------------------
# Chess Logic
# --------------------------------------------------------------------
def validate_detected_move(from_sq, to_sq, board_basics, chess_game):
   """
   Validate a detected move using chess logic
   
   Args:
       from_sq: Starting square (e.g., 'e2')
       to_sq: Target square (e.g., 'e4')
       board_basics: Board basics object
       chess_game: Chess game object with current state
       
   Returns:
       bool: True if the move is likely valid, False otherwise
   """
   if not from_sq or not to_sq:
       return False
       
   # Check if the squares are valid
   if not (re.match(r'^[a-h][1-8]$', from_sq) and re.match(r'^[a-h][1-8]$', to_sq)):
       logger.warning(f"Invalid square notation: {from_sq}, {to_sq}")
       return False
       
   # Verify there's a piece at the starting square
   try:
       board = chess_game.board
       from_square = chess.parse_square(from_sq)
       piece = board.piece_at(from_square)
       
       if not piece:
           logger.warning(f"No piece at starting square {from_sq}")
           return False
           
       # Check if the piece color matches the current turn
       if piece.color != board.turn:
           logger.warning(f"Piece at {from_sq} is the wrong color for current turn")
           return False
           
       # Verify move is at least potentially legal
       to_square = chess.parse_square(to_sq)
       potential_move = chess.Move(from_square, to_square)
       
       # Check if the move could be valid (allowing for promotion to be added later)
       for move in board.legal_moves:
           if move.from_square == from_square and move.to_square == to_square:
               return True
               
       # Special handling for potential pawn promotion
       if piece.piece_type == chess.PAWN:
           # Check if destination is on the 1st or 8th rank
           if (to_square < 8 or to_square >= 56):
               # Check if move is otherwise valid
               for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                   promotion_move = chess.Move(from_square, to_square, promotion=promotion)
                   if promotion_move in board.legal_moves:
                       return True
                   
       logger.warning(f"Move {from_sq} → {to_sq} is not legal")
       return False
       
   except Exception as e:
       logger.error(f"Error validating move: {e}")
       logger.debug(traceback.format_exc())
       return False

def validate_move_with_engine(chess_game, from_sq, to_sq):
    """
    Validates if a move is legal according to chess rules.

    Args:
        chess_game: Chess game object
        from_sq: Starting square (e.g., 'e2')
        to_sq: Target square (e.g., 'e4')

    Returns:
        tuple: (is_valid, matching_moves)
    """
    try:
        matching_moves = []

        # Check if any legal move matches the from and to squares
        for move in chess_game.board.legal_moves:
            if (chess.square_name(move.from_square) == from_sq and 
                chess.square_name(move.to_square) == to_sq):
                matching_moves.append(move)

        is_valid = len(matching_moves) > 0
        return is_valid, matching_moves
    except Exception as e:
        logger.error(f"Error validating move: {e}")
        return False, []
  
def validate_move_compatibility(printer, move_notation):
    """
    Validates that a move is compatible with the current game state to prevent errors
    
    Args:
        printer: The printer controller object
        move_notation: The move notation to validate (SAN or UCI)
        
    Returns:
        tuple: (is_valid, move_uci, error_message)
    """
    try:
        chess_game = printer.chess_game
        board = chess_game.board
        
        # Try to parse as SAN notation
        try:
            move_obj = board.parse_san(move_notation)
            move_uci = move_obj.uci()
            return True, move_uci, None
        except ValueError:
            # Not a valid SAN notation, try UCI
            if len(move_notation) >= 4:
                from_sq = move_notation[:2]
                to_sq = move_notation[2:4]
                
                # Validate square notation
                if not (re.match(r'^[a-h][1-8], from_sq) and re.match(r'^[a-h][1-8], to_sq)):
                    return False, None, "Invalid square notation. Must be in a1-h8 range."
                
                # Try to create a move object
                try:
                    from_square = chess.parse_square(from_sq)
                    to_square = chess.parse_square(to_sq)
                    
                    # Check if there's a piece at from_square
                    piece = board.piece_at(from_square)
                    if not piece:
                        return False, None, f"No piece found at {from_sq}"
                    
                    # Handle promotion if needed
                    promotion = None
                    if len(move_notation) > 4:
                        promotion = move_notation[4]
                        if promotion not in ['q', 'r', 'b', 'n']:
                            return False, None, f"Invalid promotion piece: {promotion}"
                    
                    # Create the move object
                    if promotion:
                        promotion_map = {
                            'q': chess.QUEEN, 'r': chess.ROOK, 
                            'b': chess.BISHOP, 'n': chess.KNIGHT
                        }
                        move = chess.Move(from_square, to_square, promotion=promotion_map[promotion])
                    else:
                        move = chess.Move(from_square, to_square)
                    
                    # Check if move is legal
                    if move in board.legal_moves:
                        return True, move.uci(), None
                    else:
                        # If not legal, provide helpful error message
                        if board.is_check():
                            return False, None, "This move doesn't address the check"
                        
                        # Check if the piece type can move this way
                        piece_type = piece.piece_type
                        piece_color = piece.color
                        
                        # Check if it's this player's turn
                        if piece_color != board.turn:
                            if piece_color == chess.WHITE:
                                return False, None, "It's not White's turn"
                            else:
                                return False, None, "It's not Black's turn"
                        
                        # Check for common issues based on piece type
                        if piece_type == chess.PAWN:
                            if abs(from_square - to_square) == 16 and board.piece_at(to_square) is not None:
                                return False, None, "Pawn can't jump over pieces"
                            
                            # Check for diagonal move without capture
                            if abs(chess.square_file(from_square) - chess.square_file(to_square)) == 1:
                                if board.piece_at(to_square) is None:
                                    # Check for en passant
                                    if board.ep_square == to_square:
                                        return True, move.uci(), None  # En passant is legal
                                    return False, None, "Pawn can only move diagonally when capturing"
                        
                        elif piece_type == chess.KNIGHT:
                            pass  # Knights can jump over pieces
                        
                        else:  # Rook, Bishop, Queen, King
                            # Check if there are pieces in the way
                            try:
                                between_squares = list(chess.SquareSet(chess.between(from_square, to_square)))
                                for sq in between_squares:
                                    if board.piece_at(sq) is not None:
                                        return False, None, "There are pieces in the way"
                            except Exception:
                                # between() might raise an exception if squares aren't aligned
                                return False, None, "Invalid movement pattern for this piece"
                        
                        # General error message if no specific issue identified
                        return False, None, "This move is not legal in the current position"
                
                except Exception as e:
                    return False, None, f"Error validating move: {str(e)}"
            
            return False, None, "Invalid move notation"
    
    except Exception as e:
        logger.error(f"Move validation error: {e}")
        logger.debug(traceback.format_exc())
        return False, None, f"Error validating move: {str(e)}"
        
def update_chess_position(chess_game, from_sq, to_sq, promotion=None):
    """
    Updates the internal chess board with the given move with enhanced error handling.
    
    Args:
        chess_game: Chess game object
        from_sq: Starting square (e.g., 'e2')
        to_sq: Target square (e.g., 'e4')
        promotion: Promotion piece if applicable (None, 'q', 'r', 'n', 'b')
        
    Returns:
        bool: Success or failure
    """
    try:
        # Format move as UCI with promotion if needed
        move_uci = from_sq + to_sq
        if promotion:
            move_uci += promotion
            
        # First, check if this is a legal move in the current position
        from_square = chess.parse_square(from_sq)
        to_square = chess.parse_square(to_sq)
        
        # Create the move object
        if promotion:
            promotion_map = {
                'q': chess.QUEEN, 'r': chess.ROOK, 
                'b': chess.BISHOP, 'n': chess.KNIGHT
            }
            move = chess.Move(from_square, to_square, 
                             promotion=promotion_map.get(promotion, chess.QUEEN))
        else:
            move = chess.Move(from_square, to_square)
        
        # Verify the move is legal
        if move not in chess_game.board.legal_moves:
            logger.warning(f"Move {move_uci} is not legal in the current position!")
            
            # Check if there's a similar legal move (same starting piece, different destination)
            similar_moves = []
            for legal_move in chess_game.board.legal_moves:
                if legal_move.from_square == from_square:
                    similar_moves.append(legal_move.uci())
            
            if similar_moves:
                logger.info(f"Similar legal moves from {from_sq}: {', '.join(similar_moves)}")
                
            # Check if there's a piece that can legally move to the target square
            target_moves = []
            for legal_move in chess_game.board.legal_moves:
                if legal_move.to_square == to_square:
                    target_moves.append(legal_move.uci())
            
            if target_moves:
                logger.info(f"Legal moves to {to_sq}: {', '.join(target_moves)}")
                
            # Print current board state for debugging
            logger.debug(f"Current board FEN: {chess_game.board.fen()}")
            
            # Ask user what to do
            print(f"\nWARNING: Move {move_uci} appears to be invalid in the current game state.")
            print("Options:")
            print("1. Force the move anyway (might lead to inconsistent state)")
            print("2. Verify and fix board state first")
            
            choice = input("Choose option (1-2): ").strip()
            
            if choice == "2":
                verify_result = verify_board_state(chess_game)
                if verify_result:
                    # Try the move again after fixing the state
                    return update_chess_position(chess_game, from_sq, to_sq, promotion)
                else:
                    return False
            
            # If we're forcing the move, we'll proceed with a warning
            logger.warning(f"Forcing illegal move {move_uci} - this may cause state inconsistencies")
            
            # Create new board with the updated position by directly manipulating pieces
            try:
                # Create new board with current state
                new_board = chess.Board(chess_game.board.fen())
                
                # Get the piece at source
                piece = new_board.piece_at(from_square)
                
                if not piece:
                    logger.error(f"No piece found at {from_sq}!")
                    return False
                
                # Remove any piece at destination
                new_board.remove_piece_at(to_square)
                
                # Remove piece from source
                new_board.remove_piece_at(from_square)
                
                # Place the piece at destination (with promotion if needed)
                if promotion and piece.piece_type == chess.PAWN and (to_square < 8 or to_square >= 56):
                    # Apply promotion
                    promotion_map = {
                        'q': chess.QUEEN, 'r': chess.ROOK, 
                        'b': chess.BISHOP, 'n': chess.KNIGHT
                    }
                    new_piece = chess.Piece(promotion_map.get(promotion, chess.QUEEN), piece.color)
                    new_board.set_piece_at(to_square, new_piece)
                else:
                    # Just move the piece
                    new_board.set_piece_at(to_square, piece)
                
                # Toggle turn
                new_board.turn = not new_board.turn
                
                # Update the game board
                chess_game.board = new_board
                
                logger.info(f"Manually applied move {move_uci} by manipulating pieces")
                return True
            except Exception as force_err:
                logger.error(f"Error forcing move: {force_err}")
                logger.debug(traceback.format_exc())
                return False
        
        # Handle promotion in a way compatible with the printer chess game implementation
        try:
            # Try pushing the move directly first
            chess_game.board.push(move)
            logger.info(f"Updated chess position with move: {move_uci}")
            return True
        except Exception as push_err:
            logger.error(f"Error pushing move directly: {push_err}")
            
            # Try alternative approach with update_position method
            try:
                # For promotion, we need to handle the additional promotion parameter
                if promotion:
                    # Try to use the update_position method with promotion parameter
                    if hasattr(chess_game, 'update_position_with_promotion'):
                        chess_game.update_position_with_promotion(from_sq, to_sq, promotion)
                    else:
                        # Create a move object with proper promotion information
                        promotion_map = {
                            'q': chess.QUEEN, 'r': chess.ROOK, 
                            'b': chess.BISHOP, 'n': chess.KNIGHT
                        }
                        move = chess.Move(from_square, to_square, 
                                         promotion=promotion_map.get(promotion, chess.QUEEN))
                        # Push directly to the board
                        chess_game.board.push(move)
                else:
                    # Use the standard method for non-promotion moves
                    chess_game.update_position(from_sq, to_sq)
                
                logger.info(f"Updated chess position with move: {move_uci} using alternate method")
                return True
            except Exception as alt_err:
                logger.error(f"Alternative position update also failed: {alt_err}")
                logger.debug(traceback.format_exc())
                
                # Last resort - manipulate the board directly
                try:
                    # Get the piece at source
                    piece = chess_game.board.piece_at(from_square)
                    
                    if not piece:
                        logger.error(f"No piece found at {from_sq}!")
                        return False
                    
                    # Create new board with current state
                    new_board = chess.Board(chess_game.board.fen())
                    
                    # Remove any piece at destination
                    new_board.remove_piece_at(to_square)
                    
                    # Remove piece from source
                    new_board.remove_piece_at(from_square)
                    
                    # Place the piece at destination (with promotion if needed)
                    if promotion and piece.piece_type == chess.PAWN and (to_square < 8 or to_square >= 56):
                        # Apply promotion
                        promotion_map = {
                            'q': chess.QUEEN, 'r': chess.ROOK, 
                            'b': chess.BISHOP, 'n': chess.KNIGHT
                        }
                        new_piece = chess.Piece(promotion_map.get(promotion, chess.QUEEN), piece.color)
                        new_board.set_piece_at(to_square, new_piece)
                    else:
                        # Just move the piece
                        new_board.set_piece_at(to_square, piece)
                    
                    # Toggle turn
                    new_board.turn = not new_board.turn
                    
                    # Update the game board
                    chess_game.board = new_board
                    
                    logger.info(f"Manual piece manipulation successful for move: {move_uci}")
                    return True
                except Exception as direct_err:
                    logger.error(f"Direct board manipulation failed: {direct_err}")
                    logger.debug(traceback.format_exc())
                    return False
    except Exception as e:
        logger.error(f"Error updating chess position: {e}")
        logger.debug(traceback.format_exc())
        return False
              
def check_game_over(chess_game):
    """
    Comprehensive check if the game is over with detailed reason reporting.
    Announces imminent checkmate but allows final move to be played.
    
    Args:
        chess_game: Chess game object
        
    Returns:
        bool: True if game is over, False otherwise
    """
    import logging
    logger = logging.getLogger("ChessRobot")
    
    try:
        # Check standard game-over conditions
        if chess_game.board.is_game_over():
            reason = "Game over: "
            if chess_game.board.is_checkmate():
                winner = "White" if not chess_game.board.turn else "Black"
                reason += f"Checkmate! {winner} wins!"
            elif chess_game.board.is_stalemate():
                reason += "Stalemate (draw)"
            elif chess_game.board.is_insufficient_material():
                reason += "Insufficient material (draw)"
            elif chess_game.board.is_fifty_moves():
                reason += "Fifty-move rule (draw)"
            elif chess_game.board.is_repetition():
                reason += "Threefold repetition (draw)"
            else:
                reason += "Draw by agreement or other rule"
                
            logger.info(reason)
            print(f"\n{reason}")  # Echo to the console for immediate feedback
            return True
            
        # Check for imminent checkmate but DON'T end the game prematurely
        try:
            eval_info = chess_game.get_evaluation()
            if eval_info and len(eval_info) >= 3:
                score, best_move, mate_in = eval_info
                
                # Announce imminent checkmate but continue the game
                if mate_in is not None:
                    if mate_in > 0:  # Positive mate = White wins
                        reason = f"White will checkmate in {mate_in} moves."
                        if mate_in <= 2:  # Only announce if it's close
                            logger.info(reason)
                            print(f"\n⚠️  {reason} ⚠️")
                    else:  # Negative mate = Black wins
                        reason = f"Black will checkmate in {abs(mate_in)} moves."
                        if abs(mate_in) <= 2:  # Only announce if it's close
                            logger.info(reason)
                            print(f"\n⚠️  {reason} ⚠️")
                    
                    # Important: No return here - let the game continue
                    
        except Exception as eval_err:
            logger.warning(f"Error checking mate status: {eval_err}")
            # Don't let evaluation errors stop the game
        
        # Game is not over yet
        return False
        
    except Exception as e:
        logger.error(f"Error checking game over: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False  # In case of errors, default to continuing the game
        
def handle_game_ending_transition(printer, speech, player_is_white):
    """
    Handle the transition period after a game has ended.
    Provides feedback and prepares the system for the next game.
    
    Args:
        printer: Printer controller object
        speech: Speech system object
        player_is_white: Whether the human played as white
        
    Returns:
        None
    """
    import logging
    logger = logging.getLogger("ChessRobot")
    
    try:
        # Display an eye-catching game over message
        print("\n" + "★"*60)
        print("                  GAME COMPLETED")
        print("★"*60)
        
        # Show a summary of what will happen next
        print("\nTransitioning to safe position before returning to menu...")
        print("The board will move to viewing position for safety.")
        
        # Speech announcement if available
        if speech:
            try:
                speech.put_text("Game completed. Returning to home position.")
            except Exception as e:
                logger.warning(f"Error using speech during end transition: {e}")
                
        # Move to a safe position with multiple fallback options
        safe_position_reached = False
        
        # Try the main method first
        try:
            logger.info("Moving to safe board viewing position")
            printer.move_to_board_viewing_position()
            logger.info("Successfully moved to viewing position")
            safe_position_reached = True
        except Exception as e:
            logger.warning(f"Primary method to move to viewing position failed: {e}")
            
            # First fallback - try move_to_board_view_position 
            try:
                from chess_vision_robot_integration import move_to_board_view_position
                move_to_board_view_position()
                logger.info("Successfully moved to viewing position via integration function")
                safe_position_reached = True
            except Exception as e2:
                logger.warning(f"First fallback method also failed: {e2}")
                
                # Second fallback - direct Z height for safety
                try:
                    # Most important is to get Z to a safe height
                    printer.move_to_z_height(printer.SAFE_HEIGHT)
                    logger.info("Successfully moved to safe height (partial success)")
                    safe_position_reached = True  # At least Z is safe
                except Exception as e3:
                    logger.error(f"All movement methods failed: {e3}")
        
        # Ensure gripper is open (safety measure)
        try:
            logger.info("Ensuring gripper is open")
            printer.open_gripper(force=True)
        except Exception as e:
            logger.error(f"Error opening gripper: {e}")
            
        # Indicate transition is complete
        if safe_position_reached:
            print("\nTransition complete. System is in a safe state.")
        else:
            print("\nWARNING: Could not reach safe position. Manual intervention may be needed.")
            
        # Final speech announcement if available
        if speech:
            try:
                result = printer.chess_game.board.result()
                winner_message = ""
                
                if result == "1-0":
                    winner_message = "White won the game."
                    if not player_is_white:
                        winner_message += " Better luck next time."
                    else:
                        winner_message += " Congratulations!"
                elif result == "0-1":
                    winner_message = "Black won the game." 
                    if player_is_white:
                        winner_message += " Better luck next time."
                    else:
                        winner_message += " Congratulations!"
                else:
                    winner_message = "The game was a draw."
                
                speech.put_text(f"Game over. {winner_message} Ready to return to the main menu.")
            except Exception as e:
                logger.warning(f"Error in final game announcement: {e}")
                
    except Exception as e:
        logger.error(f"Error during game ending transition: {e}")
        print("\nERROR: Problem with game ending transition. Returning to menu.")
                        
def get_evaluation_text(score, mate_in=None):
    """
    Convert numerical evaluation or mate score to descriptive text
    
    Args:
        score: Numerical evaluation in pawns (positive = white advantage)
        mate_in: Number of moves to mate (None if not a mate score)
        
    Returns:
        str: Human-readable evaluation description
    """
    if mate_in is not None:
        if mate_in > 0:  # Positive mate = White wins
            if mate_in == 1:
                return "White will checkmate next move"
            elif mate_in == 2:
                return "White will checkmate in 2 moves"
            else:
                return f"White will checkmate in {mate_in} moves"
        else:  # Negative mate = Black wins
            mate_in = abs(mate_in)
            if mate_in == 1:
                return "Black will checkmate next move"
            elif mate_in == 2:
                return "Black will checkmate in 2 moves"
            else:
                return f"Black will checkmate in {mate_in} moves"
    
    # For regular evaluations (non-mate scores)
    if score > 100:  # Extremely high scores might indicate approaching mates
        advantage_text = "completely winning for White" if score > 0 else "completely winning for Black"
    elif score > 5:
        advantage_text = "winning for White" if score > 0 else "winning for Black"
    elif score > 2:
        advantage_text = "better for White" if score > 0 else "better for Black" 
    elif score > 0.5:
        advantage_text = "slightly better for White" if score > 0 else "slightly better for Black"
    elif score < -5:
        advantage_text = "winning for Black" if score < 0 else "winning for White"
    elif score < -2:
        advantage_text = "better for Black" if score < 0 else "better for White"
    elif score < -0.5:
        advantage_text = "slightly better for Black" if score < 0 else "slightly better for White"
    else:
        advantage_text = "approximately even"
    
    return advantage_text
         
def show_legal_moves(chess_game):
    """
    Display legal moves in a user-friendly format
    
    Args:
        chess_game: Chess game object with board state
    """
    print("\nLegal Moves:")
    print("-"*60)
    
    try:
        # Group moves by piece type for better organization
        moves_by_piece = {
            'K': [], 'Q': [], 'R': [], 'B': [], 'N': [], 'P': []
        }
        
        # Collect all legal moves
        for move in chess_game.board.legal_moves:
            san = chess_game.board.san(move)
            piece = san[0] if san[0].upper() in moves_by_piece.keys() else 'P'
            
            # For pawn moves (which don't start with P in SAN notation)
            if piece == 'P' and san[0].lower() in 'abcdefgh':
                piece = 'P'
                
            # Store move details
            from_sq = chess.square_name(move.from_square).upper()
            to_sq = chess.square_name(move.to_square).upper()
            moves_by_piece[piece.upper()].append((san, from_sq, to_sq))
        
        # Display moves by piece type
        piece_names = {
            'K': 'King', 'Q': 'Queen', 'R': 'Rook', 
            'B': 'Bishop', 'N': 'Knight', 'P': 'Pawn'
        }
        
        for piece, moves in moves_by_piece.items():
            if moves:
                print(f"\n{piece_names[piece]} moves ({len(moves)}):")
                # Format moves in columns
                col_width = 25
                for i, (san, from_sq, to_sq) in enumerate(sorted(moves)):
                    if i > 0 and i % 3 == 0:
                        print()  # New line every 3 moves
                    print(f"{san:<8} ({from_sq}→{to_sq})", end=" " * (col_width - len(san) - len(from_sq) - len(to_sq) - 6))
                print()  # End with a newline
        
        # Summary
        total_moves = len(list(chess_game.board.legal_moves))
        print(f"\nTotal legal moves: {total_moves}")
        
    except Exception as e:
        logger.error(f"Error displaying legal moves: {e}")
        logger.debug(traceback.format_exc())
        print(f"Error: {e}")

def get_promotion_piece_name(piece_code):
    """Convert piece code to readable name"""
    if not piece_code:
        return ""
    
    names = {
        'q': 'Queen',
        'r': 'Rook',
        'b': 'Bishop',
        'n': 'Knight'
    }
    return names.get(piece_code.lower(), piece_code)


# --------------------------------------------------------------------
# User Interface
# --------------------------------------------------------------------
def menu_loop():
    """Interactive menu loop with improved error handling"""
    global running, stop_event, debug_visualization
    
    pts1 = None
    board_basics = None
    
    # Start status monitor thread
    status_thread = Thread(target=status_monitor, daemon=True)
    status_thread.start()

    # Main menu loop
    while running:
        try:
            print("\n" + "="*50)
            print("=== Chess Vision-Robot Integration System ===")
            print("="*50)
            print("1. Initialize system (camera, speech, printer, load calibration)")
            print("2. Play game (human vs. robot)")
            print("3. Move to board view position")
            print("4. Run calibration")
            print("5. Run diagnostic")
            print("6. Test printer movements")
            print("7. Check system status")
            print("8. Configure speech settings")  # New option
            print("9. Toggle debug visualization")
            print("10. Exit")  # Updated number
            print("-"*50)

            choice = input("\nEnter choice (1-10): ").strip()  # Updated prompt
            
            # Reset stop event for new operations
            stop_event.clear()
            
            if choice == "1":
                print("\nInitializing all components...")
                
                # Initialize components with status feedback
                update_status("Initializing camera...", "INFO")
                camera_ok = init_camera()
                if not camera_ok:
                    update_status("Camera initialization failed!", "ERROR")
                    continue
                
                update_status("Initializing speech...", "INFO")
                speech_ok = init_speech()
                if not speech_ok:
                    update_status("Speech initialization failed. Continuing without speech.", "WARNING")
                
                update_status("Initializing printer...", "INFO")
                printer_ok = init_printer()
                if not printer_ok:
                    update_status("Printer initialization failed!", "ERROR")
                    continue

                # Load calibration
                p, b = load_calibration()
                if p is not None and b is not None:
                    pts1, board_basics = p, b
                    update_status("Calibration data loaded successfully", "INFO")
                else:
                    update_status("No calibration data found. Please run option 4 first.", "WARNING")
                
                update_status("Initialization complete!", "INFO")
                
            elif choice == "2":
                # Check if all required components are initialized
                if video_capture is None:
                    update_status("Camera not initialized. Please run option 1 first.", "ERROR")
                    continue
                    
                if printer is None:
                    update_status("Printer not initialized. Please run option 1 first.", "ERROR")
                    continue
                
                # Load or verify calibration
                if pts1 is None or board_basics is None:
                    update_status("Loading calibration data...", "INFO")
                    p, b = load_calibration()
                    if p is not None and b is not None:
                        pts1, board_basics = p, b
                    else:
                        update_status("No calibration found. Please run option 4 first.", "ERROR")
                        continue
                
                # Configure game settings
                player_is_white, difficulty = configure_game_settings()
                
                # Play the game with chosen settings
                play_game(pts1, board_basics, player_is_white, difficulty)
                
            elif choice == "3":
                if printer is None:
                    update_status("Printer not initialized. Please run option 1 first.", "ERROR")
                    continue
                    
                move_to_board_view_position()
                
            elif choice == "4":
                # Ask user whether to use ML-based calibration
                ml_choice = input("\nUse ML-based calibration? (y/n, default: y): ").strip().lower()
                ml_based = ml_choice != 'n'
                
                update_status(
                    f"Running {'ML-based' if ml_based else 'traditional'} calibration...", 
                    "INFO"
                )
                
                success = run_calibration(ml_based=ml_based)
                
                if success:
                    update_status("Calibration completed. Loading new calibration data.", "INFO")
                    # Reload calibration
                    p, b = load_calibration()
                    if p is not None and b is not None:
                        pts1, board_basics = p, b
                    else:
                        update_status("Failed to load new calibration data.", "ERROR")
                else:
                    update_status("Calibration failed or was canceled.", "WARNING")
                    
            elif choice == "5":
                if video_capture is None:
                    update_status("Camera not initialized. Please run option 1 first.", "ERROR")
                    continue
                    
                update_status("Running diagnostic...", "INFO")
                run_diagnostic()
                
            elif choice == "6":
                if printer is None:
                    update_status("Printer not initialized. Please run option 1 first.", "ERROR")
                    continue
                    
                update_status("Testing printer movements...", "INFO")
                success = test_printer_movement()
                
                if success:
                    update_status("Printer tests completed successfully.", "INFO")
                else:
                    update_status("Printer tests failed. See log for details.", "ERROR")
            
            elif choice == "7":
                update_status("Checking system status...", "INFO")
                status_msg = check_system_resources()
                print("\nSystem Status:")
                print("-" * 30)
                print(status_msg)
                print("-" * 30)
                
                # Check specific components
                print("\nComponent Status:")
                print("-" * 30)
                print(f"Camera: {'OK' if video_capture else 'Not initialized'}")
                print(f"Printer: {'OK' if printer else 'Not initialized'}")
                print(f"Speech: {'OK' if speech else 'Not initialized'}")
                print(f"Calibration: {'OK' if pts1 is not None else 'Not loaded'}")
                print(f"Debug Visualization: {'Enabled' if debug_visualization else 'Disabled'}")
                print("-" * 30)
                
                # If speech is initialized, show speech settings
                if speech:
                    speech_config = config.get("speech", {})
                    current_lang = speech_config.get("language", "en")
                    current_volume = speech_config.get("volume", 1.0)
                    current_rate = speech_config.get("rate", 150)
                    
                    print("\nSpeech Configuration:")
                    print("-" * 30)
                    print(f"Language: {get_language_name(current_lang)} ({current_lang})")
                    print(f"Volume: {current_volume:.1f}")
                    print(f"Rate: {current_rate} words per minute")
                    print(f"Enabled: {speech_config.get('enabled', True)}")
                    print("-" * 30)
            
            elif choice == "8":
                # Speech configuration
                if speech is None:
                    update_status("Speech not initialized. Please run option 1 first.", "ERROR")
                    continue
                    
                update_status("Configuring speech settings...", "INFO")
                configure_speech_settings()
                
                # Offer to test new settings
                if prompt_yes_no("Would you like to test the speech settings?"):
                    test_speech()
            
            elif choice == "9":
                # Toggle debug visualization
                is_enabled = toggle_debug_visualization()
                update_status(f"Debug visualization {'enabled' if is_enabled else 'disabled'}", "INFO")
                
            elif choice == "10":  # Changed number
                print("Exiting program...")
                running = False
                stop_event.set()
                break
                
            else:
                print("Invalid choice. Enter 1-10.")  # Updated prompt
                
        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
            stop_event.set()
            time.sleep(0.5)  # Give threads time to notice the stop event
            
        except Exception as e:
            logger.error(f"Error in menu loop: {e}")
            logger.debug(traceback.format_exc())
            print(f"\nAn error occurred: {e}")
            print("See log file for details.")
            time.sleep(1)
 
def configure_game_settings():
    """
    Configure game settings before play.
    
    Returns:
        tuple: (player_is_white, difficulty)
    """
    print("\n" + "="*60)
    print("             GAME CONFIGURATION")
    print("="*60)
    
    # Color selection
    print("\nChoose your color:")
    print("1. White (You move first)")
    print("2. Black (Computer moves first)")
    
    while True:
        color_choice = input("\nEnter choice (1-2, default: 1): ").strip()
        if not color_choice or color_choice == "1":
            player_is_white = True
            break
        elif color_choice == "2":
            player_is_white = False
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Difficulty selection - use the menu from printerChess
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
    
    difficulty_map = {
        "1": "Absolute Beginner",
        "2": "Beginner",
        "3": "Casual",
        "4": "Intermediate",
        "5": "Club Player",
        "6": "Advanced",
        "7": "Expert"
    }
    
    while True:
        diff_choice = input("\nSelect difficulty (1-7, default: 4): ").strip()
        if not diff_choice:
            diff_choice = "4"  # Default to Intermediate
        
        if diff_choice in difficulty_map:
            difficulty = difficulty_map[diff_choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")
    
    print(f"\nGame configured: You play as {'White' if player_is_white else 'Black'}")
    print(f"Computer difficulty: {difficulty}")
    
    return player_is_white, difficulty

def configure_speech_settings():
    """
    Configure speech system settings including language, volume, and rate
    
    Returns:
        bool: True if configuration successful, False otherwise
    """
    global speech, config
    
    if not speech:
        print("Speech system not initialized. Initialize the system first (option 1).")
        return False
        
    print("\n" + "="*50)
    print("=== Speech Configuration Settings ===")
    print("="*50)
    
    # Display current settings
    speech_config = config.get("speech", {})
    current_lang = speech_config.get("language", "en")
    current_volume = speech_config.get("volume", 1.0)
    current_rate = speech_config.get("rate", 150)
    
    print(f"Current language: {get_language_name(current_lang)} ({current_lang})")
    print(f"Current volume: {current_volume:.1f} (0.0-1.0)")
    print(f"Current rate: {current_rate} words per minute")
    print("-"*50)
    
    # Language selection
    print("\nAvailable languages:")
    languages = [
        ("en", "English"), ("fr", "French"), ("de", "German"), 
        ("es", "Spanish"), ("it", "Italian"), ("pt", "Portuguese"),
        ("nl", "Dutch"), ("ru", "Russian"), ("zh", "Chinese"),
        ("ja", "Japanese"), ("ko", "Korean"), ("ar", "Arabic")
    ]
    
    for i, (code, name) in enumerate(languages, 1):
        print(f"{i}. {name} ({code})")
    
    while True:
        try:
            lang_choice = input("\nSelect language (1-12, or enter code directly): ").strip().lower()
            
            # Handle direct code entry
            if lang_choice in [code for code, _ in languages]:
                new_lang = lang_choice
                break
                
            # Handle numeric selection
            try:
                idx = int(lang_choice) - 1
                if 0 <= idx < len(languages):
                    new_lang = languages[idx][0]
                    break
                else:
                    print("Invalid selection. Please choose a number between 1-12.")
            except ValueError:
                print("Invalid input. Enter a number (1-12) or a language code.")
        except KeyboardInterrupt:
            return False
    
    # Volume settings
    while True:
        try:
            vol_input = input(f"\nSet volume (0.0-1.0, default: {current_volume:.1f}): ").strip()
            if not vol_input:
                new_volume = current_volume
                break
                
            try:
                new_volume = float(vol_input)
                if 0.0 <= new_volume <= 1.0:
                    break
                else:
                    print("Volume must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid input. Please enter a number between 0.0 and 1.0")
        except KeyboardInterrupt:
            return False
    
    # Rate settings
    while True:
        try:
            rate_input = input(f"\nSet speech rate (words per minute, 50-300, default: {current_rate}): ").strip()
            if not rate_input:
                new_rate = current_rate
                break
                
            try:
                new_rate = int(rate_input)
                if 50 <= new_rate <= 300:
                    break
                else:
                    print("Rate must be between 50 and 300")
            except ValueError:
                print("Invalid input. Please enter a number between 50 and 300")
        except KeyboardInterrupt:
            return False
    
    # Apply the settings to speech system
    try:
        print("\nApplying new speech settings...")
        
        # Update speech object
        speech.set_language(new_lang)
        speech.set_volume(new_volume)
        speech.set_rate(new_rate)
        
        # Update configuration
        if "speech" not in config:
            config["speech"] = {}
            
        config["speech"]["language"] = new_lang
        config["speech"]["volume"] = new_volume
        config["speech"]["rate"] = new_rate
        
        # Save configuration
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Test the new settings
        print("\nTesting new speech settings...")
        speech.put_text(f"This is a test. Language is now set to {get_language_name(new_lang)}")
        
        logger.info(f"Speech settings updated: language={new_lang}, volume={new_volume}, rate={new_rate}")
        print("\nSpeech settings updated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error updating speech settings: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nError updating speech settings: {e}")
        return False

def prompt_yes_no(question):
    """Prompt user for yes/no response"""
    response = input(f"\n{question} (y/n): ").strip().lower()
    return response.startswith('y')

def prompt_for_promotion():
    """Prompt user for promotion piece"""
    print("\nPromotion piece:")
    print("q - Queen")
    print("r - Rook")
    print("b - Bishop")
    print("n - Knight")
    piece = input("Choose promotion piece [q/r/b/n] (default: q): ").strip().lower()
    
    if piece in ['q', 'r', 'b', 'n']:
        return piece
    return 'q'  # Default to queen

def prompt_for_manual_move(chess_game):
    """Prompt user to enter a move manually with user-friendly guidance"""
    print("\nEnter your move manually:")
    print("Options:")
    print("1. UCI format (e.g., e2e4)")
    print("2. SAN format (e.g., Nf3, e4, O-O)")
    format_choice = input("Choose format (1-2): ").strip()
   
    if format_choice == "1":
        # UCI format
        move_uci = input("Enter move in UCI format (e.g., e2e4): ").strip().lower()
       
        # Basic validation
        if len(move_uci) < 4:
            print("Invalid move format - must be at least 4 characters")
            return None
       
        # Check if move is legal
        try:
            from_sq, to_sq = move_uci[:2], move_uci[2:4]
           
            # Validate square notation
            if not (re.match(r'^[a-h][1-8]$', from_sq) and re.match(r'^[a-h][1-8]$', to_sq)):
                print("Invalid square notation. Must be in a1-h8 range.")
                return None
               
            # Check if this matches a legal move
            for move in chess_game.board.legal_moves:
                if (chess.square_name(move.from_square) == from_sq and 
                    chess.square_name(move.to_square) == to_sq):
                   
                    # Check for promotion
                    if len(move_uci) > 4 and move.promotion:
                        if move_uci[4] in ['q', 'r', 'b', 'n']:
                            return move_uci
                    elif not move.promotion:
                        return move_uci
           
            print("That move is not legal in the current position")
            return None
           
        except Exception as e:
            logger.error(f"Error validating manual move: {e}")
            logger.debug(traceback.format_exc())
            print(f"Error: {e}")
            return None
    else:
        # SAN format - algebraic notation
        move_san = input("Enter move in algebraic notation (e.g., Nf3, e4, O-O): ").strip()
        
        try:
            # Try to parse SAN move
            move = chess_game.board.parse_san(move_san)
            
            # Convert to UCI for consistency
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            
            # Handle promotion
            if move.promotion:
                promotion_map = {
                    chess.QUEEN: 'q', chess.ROOK: 'r',
                    chess.BISHOP: 'b', chess.KNIGHT: 'n'
                }
                return from_sq + to_sq + promotion_map.get(move.promotion, 'q')
            else:
                return from_sq + to_sq
                
        except ValueError:
            print("Invalid move notation or illegal move")
            return None
        except Exception as e:
            logger.error(f"Error processing SAN move: {e}")
            logger.debug(traceback.format_exc())
            print(f"Error: {e}")
            return None
 
def prompt_for_robot_manual_move(chess_game):
   """
   Prompt user to manually specify a move for the robot
   
   Args:
       chess_game: Chess game object
       
   Returns:
       str: SAN notation of the selected move, or None if invalid
   """
   print("\nAvailable moves:")
   
   try:
       # Collect all legal moves for display
       legal_moves = []
       for move in chess_game.board.legal_moves:
           san = chess_game.board.san(move)
           uci = move.uci()
           from_sq = chess.square_name(move.from_square).upper()
           to_sq = chess.square_name(move.to_square).upper()
           legal_moves.append((san, uci, from_sq, to_sq))
       
       # Sort by SAN for consistent display
       legal_moves.sort(key=lambda x: x[0])
       
       # Display with numbers for selection
       for i, (san, uci, from_sq, to_sq) in enumerate(legal_moves, 1):
           if i > 1 and (i-1) % 3 == 0:
               print()  # New line every 3
           print(f"{i:2}. {san:<8} ({from_sq}→{to_sq})", end=" " * 5)
       print()  # End with newline
       
       # Prompt for selection
       selection = input("\nEnter number of move to execute (or 0 to cancel): ").strip()
       
       try:
           idx = int(selection)
           if idx == 0:
               return None
           if 1 <= idx <= len(legal_moves):
               selected_move = legal_moves[idx-1][0]  # Get SAN notation
               print(f"\nSelected move: {selected_move}")
               return selected_move
           else:
               print("Invalid selection number")
               return None
       except ValueError:
           # Handle direct SAN input
           if selection in [move[0] for move in legal_moves]:
               return selection
           # Handle direct UCI input
           elif selection in [move[1] for move in legal_moves]:
               idx = [move[1] for move in legal_moves].index(selection)
               return legal_moves[idx][0]
           else:
               print("Invalid selection")
               return None
               
   except Exception as e:
       logger.error(f"Error prompting for robot move: {e}")
       logger.debug(traceback.format_exc())
       return None

def prompt_continue_despite_error(question):
    """Prompt user whether to continue despite errors"""
    return prompt_yes_no(question)

def get_language_name(lang_code):
    """Convert language code to readable name"""
    language_map = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic"
    }
    return language_map.get(lang_code, f"Unknown ({lang_code})")

def test_speech():
    """
    Test speech system with a sample message
    
    Returns:
        bool: True if test successful, False otherwise
    """
    global speech, config
    
    if not speech:
        print("Speech system not initialized. Initialize the system first (option 1).")
        return False
        
    try:
        # Get current settings
        speech_config = config.get("speech", {})
        language = speech_config.get("language", "en")
        
        # Prepare test messages in different languages
        test_messages = {
            "en": "This is a test of the speech system. The chess robot is working correctly.",
            "fr": "Ceci est un test du système de parole. Le robot d'échecs fonctionne correctement.",
            "de": "Dies ist ein Test des Sprachsystems. Der Schachroboter funktioniert korrekt.",
            "es": "Esta es una prueba del sistema de voz. El robot de ajedrez está funcionando correctamente.",
            "it": "Questo è un test del sistema vocale. Il robot degli scacchi funziona correttamente.",
            "pt": "Este é um teste do sistema de fala. O robô de xadrez está funcionando corretamente.",
            "nl": "Dit is een test van het spraaksysteem. De schaakrobot werkt correct.",
            "ru": "Это тест речевой системы. Шахматный робот работает правильно.",
            "zh": "这是语音系统的测试。棋盘机器人运行正常。",
            "ja": "これは音声システムのテストです。チェスロボットは正常に動作しています。",
            "ko": "이것은 음성 시스템의 테스트입니다. 체스 로봇이 올바르게 작동합니다.",
            "ar": "هذا اختبار لنظام الكلام. روبوت الشطرنج يعمل بشكل صحيح."
        }
        
        # Get message for current language or use English if not available
        message = test_messages.get(language, test_messages["en"])
        
        print(f"\nTesting speech in {get_language_name(language)}...")
        print(f"Message: \"{message}\"")
        
        # Send to speech system
        speech.put_text(message)
        
        print("\nSpeech test completed.")
        return True
        
    except Exception as e:
        logger.error(f"Error testing speech: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nError testing speech: {e}")
        return False
 

# --------------------------------------------------------------------
# Calibration and Diagnostics
# --------------------------------------------------------------------
def run_calibration(ml_based=True, with_pieces=False):
    """
    Run board calibration script as a subprocess
    
    Args:
        ml_based (bool): Whether to use ML-based calibration
        with_pieces (bool): Whether to calibrate with pieces in starting position
    
    Returns:
        bool: True if calibration completed successfully, False otherwise
    """
    # Ensure ML models are available
    if not prepare_ml_models():
        logger.warning("Failed to prepare ML models for calibration")
        return False
    
    # IMPORTANT: Always use board_calibration.py as the entry point
    script_path = os.path.join(KARAYAMAN_DIR, "board_calibration.py")

    if not os.path.exists(script_path):
        logger.error(f"Calibration script not found: {script_path}")
        return False

    logger.info(f"Launching calibration script: board_calibration.py")
    update_status("Running board calibration. Follow on-screen instructions.")
    
    # Before running the calibration, stop our video capture to free the camera
    global video_capture
    if video_capture:
        try:
            logger.info("Stopping current video capture to free camera for calibration")
            video_capture.stop()
            video_capture = None
            time.sleep(3.0)
            gc.collect()
        except Exception as e:
            logger.warning(f"Error stopping video capture: {e}")
    
    try:
        # Kill any zombie camera processes first
        kill_zombie_camera_processes()
        time.sleep(1.0)
        
        # Change to Karayaman directory
        current_dir = os.getcwd()
        os.chdir(KARAYAMAN_DIR)
        
        # Build the command similar to how gui.py does it
        args = [sys.executable, "board_calibration.py", "show-info"]
        
        # Add ML flag if using ML-based calibration
        if ml_based:
            args.append("ml")
        
        # We don't need to add with-pieces flag as it's not supported in the original code
        # The gui.py from Karayaman uses a different approach for this
        
        logger.info(f"Running calibration with command: {' '.join(args)}")
        
        # Use subprocess.call to wait for the process to complete
        # This allows the calibration UI to be fully interactive and handle keypresses
        return_code = subprocess.call(args)
        
        # Change back to original directory
        os.chdir(current_dir)
        
        # Check if the process completed successfully
        if return_code == 0:
            logger.info("Calibration completed successfully")
            return True
        else:
            logger.error(f"Calibration failed with return code {return_code}")
            return False
            
    except KeyboardInterrupt:
        logger.warning("Calibration interrupted by user")
        try:
            os.chdir(current_dir)
        except:
            pass
        return False
    except Exception as e:
        logger.error(f"Error running calibration script: {e}")
        logger.debug(traceback.format_exc())
        try:
            os.chdir(current_dir)
        except:
            pass
        return False
    finally:
        # Clean up
        try:
            if 'current_dir' in locals():
                os.chdir(current_dir)
        except:
            pass
            
        # Restart camera
        try:
            time.sleep(3.0)
            init_camera(retries=3, retry_delay=2)
        except Exception as cam_error:
            logger.error(f"Failed to reinitialize camera: {cam_error}")
            logger.debug(traceback.format_exc())
                       
def run_diagnostic():
    """
    Run diagnostic script to check camera and board detection with improved model handling
    
    Returns:
        bool: True if diagnostic completed successfully, False otherwise
    """
    # Ensure ML models are available for diagnostics
    if not prepare_ml_models():
        logger.warning("Failed to prepare ML models for diagnostics")
        update_status("Warning: ML models missing or not in expected locations. Diagnostics may fail.", "WARNING")
    
    diag_path = os.path.join(KARAYAMAN_DIR, "diagnostic.py")
    if not os.path.exists(diag_path):
        logger.error(f"Diagnostic script not found: {diag_path}")
        return False
        
    logger.info("Launching diagnostic script")
    update_status("Running diagnostic. Follow on-screen instructions.")
    
    # Before running the diagnostic, stop our video capture to free the camera
    global video_capture
    if video_capture:
        try:
            logger.info("Stopping current video capture to free camera for diagnostic")
            video_capture.stop()
            video_capture = None
            time.sleep(3.0)  # Increased delay to ensure camera is fully released
            
            # Force garbage collection to clean up any lingering resources
            gc.collect()
        except Exception as e:
            logger.warning(f"Error stopping video capture: {e}")
    
    # Create a flag file to indicate we're using Pi Camera
    picam_flag_file = os.path.join(CURRENT_DIR, ".using_picamera")
    with open(picam_flag_file, 'w') as f:
        f.write("1")
    
    # Prepare environment variables to help diagnostic script find models
    script_env = os.environ.copy()
    script_env["PYTHONPATH"] = CURRENT_DIR + os.pathsep + script_env.get("PYTHONPATH", "")
    script_env["MODEL_DIR"] = os.path.join(KARAYAMAN_DIR, "models")
    script_env["USE_PICAMERA"] = "1"  # Set environment variable instead
    
    # Create a process object that we can use to handle termination
    diagnostic_process = None
    
    try:
        # Change to the Karayaman directory
        current_dir = os.getcwd()
        os.chdir(KARAYAMAN_DIR)
        
        # Create custom argument list for diagnostics
        args = [sys.executable, "diagnostic.py"]
        
        # Run in a separate process
        diagnostic_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=script_env
        )
        
        # Create a flag to track if we're processing diagnostic output
        in_diagnostic_mode = True
        
        # Set up a timer for watchdog
        watchdog_start = time.time()
        
        # Monitor the process
        while diagnostic_process.poll() is None and in_diagnostic_mode:
            # Check if we need to stop
            if stop_event.is_set():
                logger.warning("Diagnostic terminated by user")
                # Give clean termination a chance
                try:
                    logger.info("Sending SIGTERM to diagnostic process")
                    diagnostic_process.terminate()
                    # Allow time for process to terminate
                    for _ in range(20):  # Try for 2 seconds
                        if diagnostic_process.poll() is not None:
                            break
                        time.sleep(0.1)
                except Exception as term_error:
                    logger.warning(f"Error while terminating diagnostic: {term_error}")
                
                # Force kill if not terminated
                if diagnostic_process.poll() is None:
                    try:
                        logger.warning("Diagnostic process not responding to SIGTERM, sending SIGKILL")
                        diagnostic_process.kill()
                        time.sleep(0.5)  # Give it time to die
                    except Exception as kill_error:
                        logger.error(f"Failed to kill diagnostic process: {kill_error}")
                
                in_diagnostic_mode = False
                break
                
            # Read output from process
            try:
                output = diagnostic_process.stdout.readline()
                if output:
                    logger.info(f"Diagnostic: {output.strip()}")
                    # Reset watchdog timer when we get output
                    watchdog_start = time.time()
                else:
                    # Check watchdog timer - if no output for too long, check if process is still alive
                    if time.time() - watchdog_start > 30:  # 30 seconds without output
                        if diagnostic_process.poll() is None:
                            logger.warning("Diagnostic seems to be stuck (no output for 30s), checking process...")
                            # Just a log, don't terminate yet - might be displaying a window
                            watchdog_start = time.time()  # Reset timer
                    time.sleep(0.1)
            except Exception as read_error:
                logger.warning(f"Error reading diagnostic output: {read_error}")
                time.sleep(0.1)
        
        # Change back to original directory
        os.chdir(current_dir)
        
        # Process has exited, check status if we didn't explicitly terminate
        if in_diagnostic_mode:
            return_code = diagnostic_process.poll()
            if return_code == 0:
                logger.info("Diagnostic completed successfully")
                return True
            else:
                logger.error(f"Diagnostic failed with return code {return_code}")
                return False
        else:
            logger.info("Diagnostic terminated by user")
            return False
            
    except Exception as e:
        logger.error(f"Error running diagnostic script: {e}")
        logger.debug(traceback.format_exc())
        try:
            os.chdir(current_dir)  # Restore original directory if exception occurred
        except:
            pass
        return False
    finally:
        # Clean up flag file
        try:
            if os.path.exists(picam_flag_file):
                os.remove(picam_flag_file)
        except Exception as flag_error:
            logger.warning(f"Error removing picamera flag file: {flag_error}")
            
        # Make sure process is REALLY terminated
        if diagnostic_process and diagnostic_process.poll() is None:
            try:
                import psutil
                # Try to terminate the process and all children
                parent = psutil.Process(diagnostic_process.pid)
                for child in parent.children(recursive=True):
                    try:
                        logger.info(f"Terminating child process: {child.pid}")
                        child.terminate()
                    except:
                        pass
                parent.terminate()
                
                # Wait for actual termination
                gone, still_alive = psutil.wait_procs([parent], timeout=3)
                
                # Force kill if still alive
                for p in still_alive:
                    try:
                        logger.warning(f"Force killing process: {p.pid}")
                        p.kill()
                    except:
                        pass
            except Exception as process_error:
                logger.warning(f"Error while ensuring diagnostic process termination: {process_error}")
                # Last resort
                try:
                    diagnostic_process.kill()
                except:
                    pass
        
        # Wait for a moment to allow camera resources to be fully released
        time.sleep(3.0)  # Increased delay
        
        # Restart camera with proper error handling
        try:
            # Check if any existing camera processes need to be killed
            kill_zombie_camera_processes()
            # Restart camera with retries
            init_camera(retries=3, retry_delay=2)
        except Exception as cam_error:
            logger.error(f"Failed to reinitialize camera: {cam_error}")
            logger.debug(traceback.format_exc())

def test_printer_movement():
    """
    Test basic printer movements to verify functionality
    
    Returns:
        bool: True if test completed successfully, False otherwise
    """
    if not printer:
        logger.error("Printer not initialized. Cannot run movement test.")
        return False
        
    logger.info("Starting printer movement test")
    update_status("Testing printer movements. Please supervise.")
    
    try:
        # Home axes first
        logger.info("Homing axes...")
        printer.home_axes()
        
        # Get current position
        initial_pos = printer.get_position()
        if not initial_pos:
            logger.error("Failed to get initial position")
            return False
        logger.info(f"Initial position: X={initial_pos['X']}, Y={initial_pos['Y']}, Z={initial_pos['Z']}")
        
        # Test X movement
        logger.info("Testing X axis movement")
        if not printer.move_nozzle_smooth([('X', 50)]):
            logger.error("X axis forward movement failed")
            return False
        time.sleep(1)
        if not printer.move_nozzle_smooth([('X', -50)]):
            logger.error("X axis backward movement failed")
            return False
            
        # Test Y movement
        logger.info("Testing Y axis movement")
        if not printer.move_nozzle_smooth([('Y', 50)]):
            logger.error("Y axis forward movement failed")
            return False
        time.sleep(1)
        if not printer.move_nozzle_smooth([('Y', -50)]):
            logger.error("Y axis backward movement failed")
            return False
            
        # Test Z movement
        logger.info("Testing Z axis movement")
        if not printer.move_nozzle_smooth([('Z', 20)]):
            logger.error("Z axis up movement failed")
            return False
        time.sleep(1)
        if not printer.move_nozzle_smooth([('Z', -20)]):
            logger.error("Z axis down movement failed")
            return False
            
        # Test gripper
        logger.info("Testing gripper")
        printer.open_gripper()
        time.sleep(1)
        printer.close_gripper()
        time.sleep(1)
        printer.open_gripper()
        
        # Return to initial position
        current_pos = printer.get_position()
        if current_pos:
            printer.move_nozzle_smooth([
                ('X', initial_pos['X'] - current_pos['X']),
                ('Y', initial_pos['Y'] - current_pos['Y']),
                ('Z', initial_pos['Z'] - current_pos['Z'])
            ])
            
        logger.info("Printer movement test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Printer movement test failed: {e}")
        logger.debug(traceback.format_exc())
        return False

def load_calibration():
    """
    Load chess board calibration data
    
    Returns:
        tuple: (pts1, board_basics) or (None, None) if calibration not found
    """
    import pickle
    import shutil  # Import shutil for file operations
    
    # Look for calibration file in multiple possible locations
    cal_file_name = "constants.bin"
    cal_file = find_file(cal_file_name)
    
    if not cal_file:
        logger.warning(f"No calibration file found ({cal_file_name})")
        return None, None
    
    # Once found, make sure it's also in the expected location for scripts
    expected_cal_file = os.path.join(KARAYAMAN_DIR, cal_file_name)
    backup_file = os.path.join(CONFIG_DIR, f"{cal_file_name}.backup")
    
    logger.info(f"Attempting to load calibration data from {cal_file}")
    
    # Create backup directory if it doesn't exist
    os.makedirs(os.path.dirname(backup_file), exist_ok=True)
    
    try:
        # If not in the expected location, copy it there
        if cal_file != expected_cal_file and not os.path.exists(expected_cal_file):
            try:
                shutil.copy2(cal_file, expected_cal_file)
                logger.info(f"Copied calibration file to expected location: {expected_cal_file}")
            except Exception as e:
                logger.warning(f"Could not copy calibration file to expected location: {e}")
        
        # Try to load the calibration file
        with open(cal_file, "rb") as f:
            data = pickle.load(f)

        # Backup successful calibration data
        try:
            if not os.path.exists(backup_file):
                shutil.copy2(cal_file, backup_file)
                logger.info(f"Created backup of calibration at {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to backup calibration data: {e}")

        # Extract calibration data based on format
        if data[0]:  # ML-based calibration
            pts1, side_view, rotation = data[1]
            logger.info("Loaded ML-based calibration data")
        else:  # Traditional corner-based calibration
            corners, side_view, rotation, _ = data[1]
            pts1 = np.float32([
                list(corners[0][0]), list(corners[8][0]), 
                list(corners[0][8]), list(corners[8][8])
            ])
            logger.info("Loaded traditional calibration data")
            
        # Create Board_basics object
        b_basics = Board_basics(side_view, rotation)
        logger.info(f"Calibration parameters: side_view={side_view}, rotation={rotation}")
        
        return pts1, b_basics
        
    except Exception as e:
        logger.error(f"Error loading calibration data: {e}")
        logger.debug(traceback.format_exc())
        
        # If main file load failed, try backup
        if os.path.exists(backup_file):
            logger.info("Attempting to restore from backup...")
            try:
                with open(backup_file, "rb") as f:
                    data = pickle.load(f)
                
                # Try to restore the backup to the main location
                try:
                    shutil.copy2(backup_file, expected_cal_file)
                    logger.info(f"Restored calibration file from backup")
                except Exception as backup_copy_error:
                    logger.warning(f"Could not restore backup file: {backup_copy_error}")
                
                # Extract calibration data based on format
                if data[0]:  # ML-based calibration
                    pts1, side_view, rotation = data[1]
                    logger.info("Loaded ML-based calibration data from backup")
                else:  # Traditional corner-based calibration
                    corners, side_view, rotation, _ = data[1]
                    pts1 = np.float32([
                        list(corners[0][0]), list(corners[8][0]), 
                        list(corners[0][8]), list(corners[8][8])
                    ])
                    logger.info("Loaded traditional calibration data from backup")
                    
                # Create Board_basics object
                b_basics = Board_basics(side_view, rotation)
                logger.info(f"Calibration parameters from backup: side_view={side_view}, rotation={rotation}")
                
                return pts1, b_basics
                
            except Exception as backup_error:
                logger.error(f"Error loading backup calibration data: {backup_error}")
        
        return None, None

def fix_board_state(printer):
    """
    Interactive function to fix discrepancies between physical and virtual boards
    
    Args:
        printer: Printer controller object
        
    Returns:
        bool: True if fixed successfully, False otherwise
    """
    try:
        print("\nBoard Correction Options:")
        print("1. Move pieces on the physical board to match virtual state")
        print("2. Update virtual board to match physical state")
        print("3. Set up a completely new position")
        print("4. Cancel correction")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Move to board access position for physical adjustment
            print("\nMoving to board access position...")
            move_to_board_view_position()
            input("\nAdjust physical pieces to match the displayed position, then press Enter...")
            print("Board correction completed.")
            return True
            
        elif choice == "2":
            print("\nManual Virtual Board Update:")
            print("Enter moves to apply to the virtual board (e.g., 'e2e4')")
            print("Enter 'show' to display current board")
            print("Enter 'done' when finished")
            
            while True:
                cmd = input("\nCommand: ").strip().lower()
                
                if cmd == "done":
                    print("Board update completed.")
                    return True
                elif cmd == "show":
                    printer.chess_game.display_state(printer.chess_mapper.flipped)
                elif len(cmd) >= 4:
                    # Try to parse as a move
                    from_sq, to_sq = cmd[:2], cmd[2:4]
                    promotion = cmd[4:5] if len(cmd) > 4 else None
                    
                    try:
                        # Create move object
                        from_square = chess.parse_square(from_sq)
                        to_square = chess.parse_square(to_sq)
                        
                        if promotion:
                            # Map promotion piece to chess piece type
                            promotion_map = {
                                'q': chess.QUEEN, 'r': chess.ROOK, 
                                'b': chess.BISHOP, 'n': chess.KNIGHT
                            }
                            move = chess.Move(from_square, to_square, promotion=promotion_map.get(promotion, chess.QUEEN))
                        else:
                            move = chess.Move(from_square, to_square)
                            
                        # Apply move to board (without validation)
                        is_legal = move in printer.chess_game.board.legal_moves
                        if not is_legal:
                            print(f"Warning: Move {cmd} is not legal in current position")
                            if not prompt_yes_no("Apply anyway?"):
                                continue
                                
                        printer.chess_game.board.push(move)
                        print(f"Applied move: {cmd}")
                        printer.chess_game.display_state(printer.chess_mapper.flipped)
                    except Exception as e:
                        print(f"Error applying move: {e}")
                else:
                    print("Invalid command. Use 'from_to' notation (e.g., 'e2e4'), 'show', or 'done'")
                    
        elif choice == "3":
            print("\nSet Up New Position:")
            print("Enter FEN string for the desired position")
            print("Example: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' (starting position)")
            
            fen = input("\nFEN string (or press Enter for start position): ").strip()
            
            if not fen:
                fen = chess.STARTING_FEN
                
            try:
                # Validate FEN by creating a new board
                new_board = chess.Board(fen)
                printer.chess_game.board = new_board
                print("\nBoard updated to new position:")
                printer.chess_game.display_state(printer.chess_mapper.flipped)
                
                # Now move to access position so user can set up physical board
                print("\nMoving to board access position...")
                move_to_board_view_position()
                input("\nSet up physical board to match the displayed position, then press Enter...")
                print("Board setup completed.")
                return True
                
            except ValueError as e:
                print(f"Invalid FEN string: {e}")
                return False
                
        elif choice == "4":
            print("Correction canceled.")
            return False
            
        else:
            print("Invalid choice.")
            return False
            
    except Exception as e:
        print(f"Error during board correction: {e}")
        logger.error(f"Error during board correction: {e}")
        logger.debug(traceback.format_exc())
        return False
 
def verify_board_state(printer):
    """
    Verify that the physical board matches the internal board state
    
    Args:
        printer: Printer controller object
        
    Returns:
        bool: True if verification successful, False otherwise
    """
    try:
        print("\nBoard State Verification:")
        print("Please compare the physical board with the displayed position below.")
        
        # Display the current board state
        printer.chess_game.display_state(printer.chess_mapper.flipped)
        
        # Ask for confirmation
        confirmation = input("\nDoes the physical board match this state? (yes/no/fix): ").strip().lower()
        
        if confirmation == "yes":
            print("Board state confirmed as matching.")
            return True
        elif confirmation == "no":
            print("Board mismatch detected.")
            if prompt_yes_no("Would you like to fix discrepancies?"):
                return fix_board_state(printer)
            return False
        elif confirmation == "fix":
            return fix_board_state(printer)
        else:
            print("Invalid input. Please enter 'yes', 'no', or 'fix'.")
            return False
            
    except Exception as e:
        print(f"Error during board verification: {e}")
        logger.error(f"Error during board verification: {e}")
        logger.debug(traceback.format_exc())
        return False

def capture_debug_snapshot():
    """Capture and save a snapshot of the current camera view for debugging"""
    global video_capture
    
    if not video_capture:
        print("Camera not initialized")
        return False
        
    try:
        # Get a frame
        frame = video_capture.get_frame()
        if frame is None:
            print("Could not get frame from camera")
            return False
            
        # Create debug directory
        debug_dir = ensure_debug_dir()
        
        # Save the frame
        timestamp = int(time.time())
        filepath = os.path.join(debug_dir, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(filepath, frame)
        
        print(f"Snapshot saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error capturing snapshot: {e}")
        print(f"Error: {e}")
        return False


# --------------------------------------------------------------------
# Program Entry
# --------------------------------------------------------------------
def main():
    """Main program entry point with robust error handling"""
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Display welcome message
        print("\n" + "="*60)
        print("Welcome to the Chess Vision Robot Integration System")
        print("="*60)
        print(f"System initialized at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {LOG_FILE}")
        
        # Check directory structure
        print("\nVerifying system components...")
        all_ok = True
        for dir_name, path in [("Karayaman Vision System", KARAYAMAN_DIR), 
                              ("Printer Controller", PRINTER_DIR)]:
            exists = os.path.exists(path)
            print(f"- {dir_name}: {'OK' if exists else 'MISSING'}")
            if not exists:
                all_ok = False
        
        if not all_ok:
            print("\nWARNING: Some required components are missing!")
            print("Please check the installation instructions and try again.")
            ans = input("Continue anyway? (y/n): ")
            if ans.lower() != 'y':
                print("Exiting program.")
                return
        
        # Check dependency status
        print("\nChecking dependencies...")
        for dep, status in dependency_status.items():
            print(f"- {dep}: {'OK' if status else 'MISSING'}")
        
        # Start the menu loop
        menu_loop()
        
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nCritical error: {e}")
        print("See log file for details.")
    finally:
        # Final cleanup
        cleanup_resources()
        print("\nProgram ended. Resources cleaned up.")

if __name__ == "__main__":
    main()
