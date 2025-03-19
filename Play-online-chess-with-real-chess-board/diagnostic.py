#!/usr/bin/env python3
"""
Pi Camera Chess Board Diagnostic Tool

This tool provides visual feedback on chess piece detection using the Raspberry Pi Camera module.
It's specifically designed to work with the Video_capture_thread class for reliable camera handling.
"""

import cv2
import numpy as np
import pickle
import time
import os
import sys
import tkinter as tk
from tkinter import messagebox
import traceback
import argparse
import logging
from datetime import datetime
import signal
import gc

# Import our custom Pi Camera thread handler
from videocapture import Video_capture_thread

# Local imports with error handling
try:
    from board_calibration_machine_learning import detect_board
    from helper import perspective_transform, predict, euclidean_distance
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you have all required modules installed.")
    sys.exit(1)

# Version information
__version__ = "1.1.0"

# Configure logging
log_dir = os.path.join(os.path.expanduser("~"), ".chess_robot", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pi_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PiChessDiagnostic")

# Constants
DETECTION_INTERVAL_MS = 100  # Interval for UI updates
PIECE_CONFIDENCE_THRESHOLD = 0.6  # Threshold for piece detection
CAMERA_WARMUP_TIME = 3  # seconds
MAX_CALIBRATION_ATTEMPTS = 100
CIRCLE_RADIUS = 10  # For displaying detected pieces

class PiChessDiagnostic:
    """Diagnostic tool for chess piece detection using Pi Camera"""
    
    def __init__(self):
        """Initialize the diagnostic tool"""
        # Camera thread
        self.video_capture = None
        
        # ML models
        self.corner_model = None
        self.piece_model = None
        self.color_model = None
        
        # Calibration data
        self.pts1 = None
        self.side_view_compensation = None
        self.rotation_count = None
        self.roi_mask = None
        
        # UI state
        self.debug_mode = False
        self.pause_display = False
        self.show_coordinates = False
        self.show_confidence = False
        self.running = True
        self.save_dir = "diagnostics"
        
        # Performance tracking
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        # Current display data
        self.current_frame = None
        self.current_processed_frame = None
        self.current_detections = []  # List of detected pieces
        
        # Parse command line arguments
        self.args = self.parse_arguments()
        
        # Setup based on arguments
        if self.args.debug:
            logger.setLevel(logging.DEBUG)
            self.debug_mode = True
            logger.debug("Debug mode enabled")
            
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='Pi Camera Chess Board Diagnostic Tool',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Diagnostic options
        parser.add_argument('--calibrate', action='store_true', 
                          help='Run calibration before diagnostic')
        parser.add_argument('--debug', action='store_true', 
                          help='Enable debug mode with extra information')
        parser.add_argument('--save-dir', type=str, default='diagnostics', 
                          help='Directory to save diagnostic images')
        parser.add_argument('--threshold', type=float, default=PIECE_CONFIDENCE_THRESHOLD,
                          help='Confidence threshold for piece detection (0-1)')
        
        # Handle old-style arguments for backward compatibility
        args, unknown = parser.parse_known_args()
        
        for arg in unknown:
            if arg == "calibrate":
                args.calibrate = True
            elif arg == "debug":
                args.debug = True
                
        return args
    
    def initialize(self):
        """Initialize camera, models, and calibration data"""
        try:
            # Load machine learning models
            logger.info("Loading ML models...")
            if not self.load_ml_models():
                return False
            
            # Initialize Pi Camera
            logger.info("Initializing Pi Camera...")
            if not self.initialize_camera():
                return False
            
            # Load or perform calibration
            if self.args.calibrate:
                logger.info("Running board calibration...")
                if not self.run_calibration():
                    logger.error("Board calibration failed")
                    self.show_error_message("Calibration Failed", 
                                          "Could not detect the chess board.\n"
                                          "Please check lighting and camera position.")
                    return False
            else:
                logger.info("Loading calibration data...")
                if not self.load_calibration():
                    logger.error("Loading calibration data failed")
                    self.show_error_message("Calibration Data Error", 
                                          "Could not load calibration data.\n"
                                          "Please run calibration first.")
                    return False
                
            logger.info("Initialization completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.debug(traceback.format_exc())
            self.show_error_message("Initialization Error", f"Failed to initialize:\n{str(e)}")
            return False
    
    def load_ml_models(self):
        """Load the machine learning models for board and piece detection"""
        try:
            # Define model files
            model_files = {
                "corner": "yolo_corner.onnx",
                "piece": "cnn_piece.onnx",
                "color": "cnn_color.onnx"
            }
            
            # Check if model files exist
            for name, file in model_files.items():
                if not os.path.exists(file):
                    logger.error(f"Model file not found: {file}")
                    self.show_error_message("Missing Model File", 
                                          f"The model file '{file}' could not be found.")
                    return False
            
            # Load the models
            self.corner_model = cv2.dnn.readNetFromONNX(model_files["corner"])
            self.piece_model = cv2.dnn.readNetFromONNX(model_files["piece"])
            self.color_model = cv2.dnn.readNetFromONNX(model_files["color"])
            
            logger.info("ML models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            logger.debug(traceback.format_exc())
            self.show_error_message("Model Loading Error", 
                                  f"Failed to load ML models: {str(e)}")
            return False
            
    def initialize_camera(self):
        """Initialize the Pi Camera with Video_capture_thread"""
        try:
            logger.info("Starting Pi Camera thread...")
            self.video_capture = Video_capture_thread()
            self.video_capture.daemon = True
            self.video_capture.start()
            
            # Wait for camera to initialize
            logger.info(f"Waiting {CAMERA_WARMUP_TIME}s for camera to initialize...")
            time.sleep(CAMERA_WARMUP_TIME)
            
            # Test camera connection
            test_frame = self.video_capture.get_frame()
            if test_frame is None:
                logger.error("Could not get frame from Pi Camera")
                self.show_error_message("Camera Error", 
                                      "Could not get frame from Pi Camera.\n"
                                      "Please check your camera connection and permissions.")
                return False
                
            logger.info(f"Pi Camera initialized successfully: {test_frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Pi Camera: {e}")
            logger.debug(traceback.format_exc())
            self.show_error_message("Camera Error", 
                                  f"Failed to initialize Pi Camera: {str(e)}")
            return False
    
    def run_calibration(self):
        """Run the board calibration process"""
        logger.info("Starting board calibration...")
        
        # Show a dialog to inform the user
        self.show_info_dialog(
            "Calibration",
            "Looking for chess board. Please ensure the board is visible, "
            "well-lit, and all four corners are in the camera view."
        )
        
        is_detected = False
        progress_interval = MAX_CALIBRATION_ATTEMPTS // 10
        
        # Try to detect the board
        for attempt in range(MAX_CALIBRATION_ATTEMPTS):
            # Show progress periodically
            if attempt % progress_interval == 0:
                logger.info(f"Calibration attempt {attempt+1}/{MAX_CALIBRATION_ATTEMPTS}...")
                
            # Get a frame from the camera
            frame = self.video_capture.get_frame()
            if frame is None:
                logger.warning("Error reading frame during calibration")
                time.sleep(0.1)
                continue
                
            # Display frame with overlay to show progress
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"Calibrating... {attempt+1}/{MAX_CALIBRATION_ATTEMPTS}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("Chess Board Calibration", display_frame)
            cv2.waitKey(1)
            
            # Try to detect the board
            try:
                result = detect_board(frame, self.corner_model, self.piece_model, self.color_model)
                if result:
                    self.pts1, self.side_view_compensation, self.rotation_count = result
                    is_detected = True
                    logger.info("Chess board detected successfully!")
                    
                    # Save calibration data
                    self.save_calibration_data()
                    
                    # Save a preview of the detected board
                    self.save_calibration_preview(frame)
                    break
            except Exception as e:
                logger.error(f"Error during board detection: {e}")
                time.sleep(0.1)
                
        # Clean up display window
        cv2.destroyAllWindows()
        
        if not is_detected:
            logger.error("Could not detect chess board after multiple attempts")
            return False
            
        return True
        
    def save_calibration_data(self):
        """Save calibration data to file"""
        try:
            filename = 'constants.bin'
            
            # Create backup if existing file
            if os.path.exists(filename):
                backup_name = f"{filename}.bak"
                try:
                    import shutil
                    shutil.copy2(filename, backup_name)
                    logger.info(f"Created backup of previous calibration data: {backup_name}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Save the new calibration data
            with open(filename, 'wb') as outfile:
                pickle.dump([True, [self.pts1, self.side_view_compensation, self.rotation_count]], outfile)
                
            logger.info(f"Calibration data saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
            
    def save_calibration_preview(self, frame):
        """Save a preview of the calibrated board"""
        try:
            # Create a copy of the frame with visual indicators
            preview = frame.copy()
            
            # Draw the board outline
            for i in range(4):
                pt1 = (int(self.pts1[i][0]), int(self.pts1[i][1]))
                pt2 = (int(self.pts1[(i+1)%4][0]), int(self.pts1[(i+1)%4][1]))
                cv2.line(preview, pt1, pt2, (0, 255, 0), 2)
            
            # Add calibration information text
            cv2.putText(
                preview, 
                f"Side view: {self.side_view_compensation}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            cv2.putText(
                preview, 
                f"Rotation: {self.rotation_count}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
                       
            # Save the preview
            preview_path = os.path.join(self.save_dir, "calibration_preview.jpg")
            cv2.imwrite(preview_path, preview)
            logger.info(f"Calibration preview saved to {preview_path}")
            
        except Exception as e:
            logger.error(f"Error saving calibration preview: {e}")
            
    def load_calibration(self):
        """Load calibration data from file"""
        try:
            filename = 'constants.bin'
            if not os.path.exists(filename):
                # Check for backup file
                backup_file = f"{filename}.bak"
                if os.path.exists(backup_file):
                    logger.warning(f"Primary calibration file not found, using backup: {backup_file}")
                    filename = backup_file
                else:
                    logger.error(f"Calibration file not found: {filename}")
                    return False
                
            with open(filename, 'rb') as infile:
                try:
                    calibration_data = pickle.load(infile)
                except Exception as e:
                    logger.error(f"Error unpickling calibration data: {e}")
                    return False
            
            # Extract calibration data
            try:
                if calibration_data[0]:
                    # ML-based calibration
                    self.pts1, self.side_view_compensation, self.rotation_count = calibration_data[1]
                    self.roi_mask = None
                    logger.info(f"Loaded ML-based calibration data (rotation: {self.rotation_count})")
                else:
                    # Traditional calibration
                    corners, self.side_view_compensation, self.rotation_count, self.roi_mask = calibration_data[1]
                    self.pts1 = np.float32([list(corners[0][0]), list(corners[8][0]), list(corners[0][8]),
                                          list(corners[8][8])])
                    logger.info(f"Loaded traditional calibration data (rotation: {self.rotation_count})")
                
                return True
            except Exception as e:
                logger.error(f"Invalid calibration data format: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
    
    def process_frame(self, frame):
        """Process the frame to detect and highlight chess pieces"""
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to process_frame")
            return np.zeros((480, 480, 3), dtype=np.uint8)
            
        try:
            # Create a copy for drawing
            processed = frame.copy()
            
            # Clear previous detections
            self.current_detections = []
            
            # Process each square on the board
            height, width = frame.shape[:2]
            for row in range(8):
                for column in range(8):
                    # Calculate square boundaries
                    minX = int(column * width / 8)
                    maxX = int((column + 1) * width / 8)
                    minY = int(row * height / 8)
                    maxY = int((row + 1) * height / 8)
                    
                    # Extract the square image
                    try:
                        square_image = frame[minY:maxY, minX:maxX]
                        
                        # Skip if square is too small
                        if square_image.shape[0] < 5 or square_image.shape[1] < 5:
                            continue
                        
                        # Calculate center of the square
                        centerX = int((minX + maxX) / 2)
                        centerY = int((minY + maxY) / 2)
                        
                        # Draw grid lines if showing coordinates
                        if self.show_coordinates:
                            # Draw square outline
                            cv2.rectangle(processed, (minX, minY), (maxX, maxY), (128, 128, 128), 1)
                            
                            # Add square name (e.g., "e4")
                            square_name = self.get_square_name(row, column)
                            cv2.putText(
                                processed,
                                square_name,
                                (minX + 5, maxY - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (255, 255, 255),
                                1
                            )
                        
                        # Detect piece with confidence
                        is_piece = predict(square_image, self.piece_model)
                        
                        if is_piece:
                            # Detect if white or black piece
                            is_white = predict(square_image, self.color_model)
                            
                            # Store detection
                            self.current_detections.append({
                                'row': row,
                                'column': column,
                                'square': self.get_square_name(row, column),
                                'is_white': bool(is_white),
                                'center': (centerX, centerY)
                            })
                            
                            # Draw a circle based on piece color
                            if is_white:
                                # Blue circle for white pieces
                                cv2.circle(processed, (centerX, centerY), CIRCLE_RADIUS, (255, 0, 0), 2)
                            else:
                                # Green circle for black pieces
                                cv2.circle(processed, (centerX, centerY), CIRCLE_RADIUS, (0, 255, 0), 2)
                    except Exception as e:
                        logger.error(f"Error processing square ({row}, {column}): {e}")
            
            # Add information overlay
            self.add_info_overlay(processed)
            
            return processed
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            # Return original frame on error
            return frame
    
    def get_square_name(self, row, column):
        """Convert row and column to chess square name (a1, b2, etc.)"""
        # Adjust for rotation count
        if self.rotation_count == 0:
            adj_row, adj_col = row, column
        elif self.rotation_count == 1:
            adj_row, adj_col = column, 7-row
        elif self.rotation_count == 2:
            adj_row, adj_col = 7-row, 7-column
        elif self.rotation_count == 3:
            adj_row, adj_col = 7-column, row
        else:
            adj_row, adj_col = row, column
            
        # Convert to chess notation
        file_letter = chr(ord('a') + adj_col)
        rank_number = 8 - adj_row
        
        return f"{file_letter}{rank_number}"
            
    def add_info_overlay(self, frame):
        """Add information overlay to the frame"""
        try:
            # Add FPS counter
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Add piece count
            white_count = sum(1 for p in self.current_detections if p["is_white"])
            black_count = sum(1 for p in self.current_detections if not p["is_white"])
            
            cv2.putText(
                frame,
                f"Pieces: {white_count+black_count} (W:{white_count} B:{black_count})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Show calibration info
            cv2.putText(
                frame,
                f"Rotation: {self.rotation_count} | Side view: {self.side_view_compensation}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Add controls help
            cv2.putText(
                frame,
                "Controls: 's':save 'q':quit 'p':pause 'c':coords 'd':debug",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show pause indicator if paused
            if self.pause_display:
                cv2.putText(
                    frame,
                    "PAUSED",
                    (frame.shape[1]//2 - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
        except Exception as e:
            logger.error(f"Error adding info overlay: {e}")
    
    def run(self):
        """Main diagnostic loop"""
        # Register signal handlers for clean shutdown
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
        except:
            # Windows doesn't support all signals
            pass
            
        # Show information dialog
        self.show_info_dialog(
            "Chess Board Diagnostic",
            "The diagnostic process will start.\n\n"
            "White pieces will be marked with blue circles.\n"
            "Black pieces will be marked with green circles.\n\n"
            "Controls:\n"
            "  's' - Save diagnostic image\n"
            "  'p' - Pause/resume display\n"
            "  'c' - Toggle square coordinates\n"
            "  'd' - Toggle debug information\n"
            "  'q' - Quit"
        )
        
        logger.info("Starting diagnostic display...")
        
        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        fps_update_interval = 30  # Update FPS every 30 frames
        
        # Main loop
        while self.running:
            try:
                # Get frame if not paused
                if not self.pause_display:
                    # Get a frame from the Pi Camera
                    original_frame = self.video_capture.get_frame()
                    
                    if original_frame is None:
                        logger.warning("Error reading frame")
                        time.sleep(0.1)
                        continue
                        
                    # Transform the frame to get a top-down view of the board
                    try:
                        transformed_frame = perspective_transform(original_frame, self.pts1)
                        
                        # Store the current frame
                        self.current_frame = transformed_frame
                        
                        # Process the frame to highlight chess pieces
                        self.current_processed_frame = self.process_frame(self.current_frame)
                        
                        # Calculate FPS
                        frame_count += 1
                        if frame_count % fps_update_interval == 0:
                            current_time = time.time()
                            elapsed_time = current_time - start_time
                            if elapsed_time > 0:
                                self.fps = fps_update_interval / elapsed_time
                                start_time = current_time
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        time.sleep(0.1)
                        continue
                
                # Skip rendering if no frame available
                if self.current_frame is None or self.current_processed_frame is None:
                    time.sleep(0.1)
                    continue
                    
                # Display the processed and original frames side by side
                display_frame = np.hstack((self.current_processed_frame, self.current_frame))
                
                # Display the frame
                cv2.imshow('Pi Chess Board Diagnostic', display_frame)
                
                # Process key presses
                key = cv2.waitKey(DETECTION_INTERVAL_MS) & 0xFF
                self.process_keypress(key)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(0.1)
                
            # Periodic garbage collection
            if frame_count % 300 == 0:  # Every 300 frames
                gc.collect()
                
        # Clean up
        self.cleanup()
            
    def process_keypress(self, key):
        """Process key presses"""
        if key == ord('q'):
            # Quit
            logger.info("User requested exit")
            self.running = False
            
        elif key == ord('s'):
            # Save diagnostic image
            self.save_diagnostic_image()
            
        elif key == ord('p'):
            # Pause/resume display
            self.pause_display = not self.pause_display
            logger.info(f"Display {'paused' if self.pause_display else 'resumed'}")
            
        elif key == ord('c'):
            # Toggle coordinates
            self.show_coordinates = not self.show_coordinates
            logger.info(f"Coordinates display {'enabled' if self.show_coordinates else 'disabled'}")
            
        elif key == ord('d'):
            # Toggle debug mode
            self.debug_mode = not self.debug_mode
            logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
            logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
            
    def save_diagnostic_image(self):
        """Save current diagnostic image"""
        try:
            # Create a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            
            # Check if we have a valid frame
            if self.current_frame is None or self.current_processed_frame is None:
                logger.warning("No valid frame to save")
                return
                
            # Create a combined image
            combined = np.hstack((self.current_processed_frame, self.current_frame))
            
            # Save the image
            cv2.imwrite(filepath, combined)
            logger.info(f"Diagnostic image saved as {filepath}")
            
            # Save a CSV with detection results for reference
            if self.current_detections:
                csv_path = os.path.join(self.save_dir, f"diagnostic_{timestamp}.csv")
                try:
                    with open(csv_path, 'w') as f:
                        f.write("square,color\n")
                        for piece in self.current_detections:
                            color = "white" if piece["is_white"] else "black"
                            f.write(f"{piece['square']},{color}\n")
                    logger.info(f"Detection data saved to {csv_path}")
                except Exception as e:
                    logger.error(f"Error saving CSV: {e}")
            
            # Show confirmation to user
            self.show_info_dialog("Save Complete", f"Diagnostic image saved to:\n{filepath}")
            
        except Exception as e:
            logger.error(f"Error saving diagnostic image: {e}")
            
    def show_info_dialog(self, title, message):
        """Show information dialog"""
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title, message)
            root.destroy()
        except Exception as e:
            logger.error(f"Error showing info dialog: {e}")
            print(f"\n{title}: {message}\n")  # Fallback to console
            
    def show_error_message(self, title, message):
        """Show error message dialog"""
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(title, message)
            root.destroy()
        except Exception as e:
            logger.error(f"Error showing error dialog: {e}")
            print(f"\nERROR - {title}: {message}\n")  # Fallback to console
            
    def signal_handler(self, sig, frame):
        """Handle termination signals gracefully"""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
        
    def cleanup(self):
        """Clean up resources"""
        logger.info("Shutting down diagnostic...")
        
        # Stop the camera thread
        if self.video_capture is not None:
            try:
                self.video_capture.stop()
                logger.info("Pi Camera stopped")
            except Exception as e:
                logger.error(f"Error stopping Pi Camera: {e}")
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Diagnostic completed")


def main():
    """Main entry point"""
    print("Pi Camera Chess Board Diagnostic Tool")
    print("-------------------------------------")
    
    # Create and run the diagnostic tool
    diagnostic = PiChessDiagnostic()
    
    try:
        if diagnostic.initialize():
            diagnostic.run()
        else:
            logger.error("Initialization failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        diagnostic.show_error_message(
            "Unexpected Error", 
            f"An unexpected error occurred:\n{str(e)}"
        )
    finally:
        # Ensure cleanup happens even if there's an exception
        try:
            diagnostic.cleanup()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    main()
