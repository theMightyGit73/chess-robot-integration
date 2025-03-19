import cv2
import platform
import pickle
import time
import os
import numpy as np
import sys
import tkinter as tk
from tkinter import messagebox

from board_calibration_machine_learning import detect_board
from helper import rotateMatrix, perspective_transform, edge_detection, euclidean_distance
from videocapture import Video_capture_thread

# Global variables for debug visualization
DEBUG_MODE = True
FRAME_COUNT = 0
LAST_LOG_TIME = time.time()

def debug_log(message):
    """Print timestamped debug messages"""
    if DEBUG_MODE:
        print(f"[DEBUG {time.time():.3f}] {message}")

def draw_calibration_overlay(frame, found_board=False, fps=0, detection_info=None):
    """Add informative overlay to the calibration window"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw a border
    cv2.rectangle(overlay, (10, 10), (w-10, h-10), (0, 255, 0) if found_board else (0, 0, 255), 3)
    
    # Add text
    status = "CHESS BOARD DETECTED" if found_board else "SEARCHING FOR CHESS BOARD"
    cv2.putText(overlay, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0) if found_board else (0, 0, 255), 2)
    
    # Add FPS counter
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
                
    # Add frame dimensions
    cv2.putText(overlay, f"Frame: {frame.shape[1]}x{frame.shape[0]}", (20, h-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add detection info if available
    if detection_info:
        cv2.putText(overlay, detection_info, (20, h-90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # Add instructions
    if found_board:
        cv2.putText(overlay, "Press 'r' to rotate, 'q' to accept", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(overlay, "Press 'q' to quit, 'd' to toggle debug view", (20, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay

def enhance_frame_for_detection(frame):
    """Apply preprocessing to improve corner detection"""
    debug_log("Enhancing frame for detection")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Create debug visualization
    if DEBUG_MODE:
        debug_frames = np.hstack((gray, enhanced, binary))
        cv2.imshow('Debug Processing', debug_frames)
    
    return gray, binary

def mark_corners(frame, augmented_corners, rotation_count):
    """Mark board corners with coordinates on the frame"""
    height, width = frame.shape[:2]
    if rotation_count == 1:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_count == 2:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_count == 3:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    for i in range(len(augmented_corners)):
        for j in range(len(augmented_corners[i])):
            if rotation_count == 0:
                index = str(i) + "," + str(j)
                corner = augmented_corners[i][j]
            elif rotation_count == 1:
                index = str(j) + "," + str(8 - i)
                corner = (height - augmented_corners[i][j][1], augmented_corners[i][j][0])
            elif rotation_count == 2:
                index = str(8 - i) + "," + str(8 - j)
                corner = (width - augmented_corners[i][j][0], height - augmented_corners[i][j][1])
            elif rotation_count == 3:
                index = str(8 - j) + "," + str(i)
                corner = (augmented_corners[i][j][1], width - augmented_corners[i][j][0])
            corner = (int(corner[0]), int(corner[1]))
            frame = cv2.putText(frame, index, corner, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # Draw a circle at each corner for better visibility
            cv2.circle(frame, corner, 5, (0, 0, 255), -1)

    return frame

def visualize_ml_detection(frame, corners_result):
    """Visualize the ML-based corner detection process"""
    if corners_result is None:
        return frame
        
    pts1, side_view_compensation, rotation_count = corners_result
    debug_frame = frame.copy()
    
    # Draw the detected corners
    for i, point in enumerate(pts1):
        cv2.circle(debug_frame, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)
        cv2.putText(debug_frame, f"{i}", (int(point[0])+10, int(point[1])+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw the board outline
    for i in range(4):
        pt1 = (int(pts1[i][0]), int(pts1[i][1]))
        pt2 = (int(pts1[(i+1)%4][0]), int(pts1[(i+1)%4][1]))
        cv2.line(debug_frame, pt1, pt2, (0, 255, 0), 2)
    
    # Add side view compensation info
    cv2.putText(
        debug_frame, 
        f"Side view comp: {side_view_compensation}", 
        (20, frame.shape[0]-120), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )
    
    # Add rotation info
    cv2.putText(
        debug_frame, 
        f"Rotation: {rotation_count}", 
        (20, frame.shape[0]-150), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )
    
    return debug_frame

def main():
    """Main function for board calibration with enhanced debugging"""
    global FRAME_COUNT, LAST_LOG_TIME
    
    # Setup file names and load models
    filename = 'constants.bin'
    corner_model = cv2.dnn.readNetFromONNX("yolo_corner.onnx")
    piece_model = cv2.dnn.readNetFromONNX("cnn_piece.onnx")
    color_model = cv2.dnn.readNetFromONNX("cnn_color.onnx")
    
    # Process command line arguments
    webcam_width = None
    webcam_height = None
    fps = None
    is_machine_learning = False
    show_info = False
    cap_index = 0
    cap_api = cv2.CAP_ANY
    platform_name = platform.system()
    
    for argument in sys.argv:
        if argument == "show-info":
            show_info = True
        elif argument.startswith("cap="):
            cap_index = int("".join(c for c in argument if c.isdigit()))
            if platform_name == "Darwin":
                cap_api = cv2.CAP_AVFOUNDATION
            elif platform_name == "Linux":
                cap_api = cv2.CAP_V4L2
            else:
                cap_api = cv2.CAP_DSHOW
        elif argument == "ml":
            is_machine_learning = True
        elif argument.startswith("width="):
            webcam_width = int(argument[len("width="):])
        elif argument.startswith("height="):
            webcam_height = int(argument[len("height="):])
        elif argument.startswith("fps="):
            fps = int(argument[len("fps="):])
    
    # Show initial info message
    if show_info:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Board Calibration",
                        'Board calibration will start. It should detect corners of the chess board almost immediately. '
                        'If it does not, you should press key "q" to stop board calibration and change camera/board position.')
    
    # Initialize Pi Camera
    debug_log("Initializing Pi Camera...")
    video_capture_thread = Video_capture_thread()
    video_capture_thread.daemon = True
    video_capture_thread.start()
    
    debug_log("Waiting for camera to initialize...")
    time.sleep(3)  # Give camera time to initialize
    
    # Basic camera check
    try:
        debug_log("Testing camera connection...")
        test_frame = video_capture_thread.get_frame()
        if test_frame is None:
            print("Couldn't get frame from camera. Please check your camera connection.")
            video_capture_thread.stop()
            sys.exit(0)
        else:
            debug_log(f"Camera test successful: frame shape = {test_frame.shape}")
    except Exception as e:
        print(f"Camera error: {e}")
        video_capture_thread.stop()
        sys.exit(0)
    
    # Define board dimensions for corner detection
    board_dimensions = (7, 7)
    
    # Warm up - take a few frames to stabilize camera
    debug_log("Camera warm-up...")
    for _ in range(5):
        frame = video_capture_thread.get_frame()
        if frame is None:
            debug_log("Error reading frame. Please check your camera connection.")
            time.sleep(0.5)
            continue
    
    # Variables for FPS calculation
    frame_times = []
    fps_value = 0
    toggle_debug_view = False
    detection_info = None

    # Main calibration loop
    debug_log("Starting chess board detection...")
    while True:
        loop_start = time.time()
        
        # Get frame with timeout
        frame = video_capture_thread.get_frame()
        if frame is None:
            debug_log("Error reading frame. Please check your camera connection.")
            time.sleep(0.5)
            continue
        
        # Calculate FPS
        FRAME_COUNT += 1
        current_time = time.time()
        frame_times.append(current_time)
        
        # Keep only the last 30 frames for FPS calculation
        if len(frame_times) > 30:
            frame_times.pop(0)
        
        if len(frame_times) > 1:
            fps_value = len(frame_times) / (frame_times[-1] - frame_times[0])
        
        # Log status every 5 seconds
        if current_time - LAST_LOG_TIME > 5:
            debug_log(f"FPS: {fps_value:.1f}, Frame size: {frame.shape}")
            LAST_LOG_TIME = current_time
        
        # Prepare debugging info
        detection_info = None
        
        # Try ML-based calibration if selected
        if is_machine_learning:
            try:
                debug_log("Attempting ML-based board detection...")
                result = detect_board(frame, corner_model, piece_model, color_model)
                
                if result:
                    pts1, side_view_compensation, rotation_count = result
                    detection_info = f"ML Detection: OK - Side view: {side_view_compensation}, Rotation: {rotation_count}"
                    
                    # Show ML detection visualization
                    if toggle_debug_view:
                        debug_frame = visualize_ml_detection(frame, result)
                        cv2.imshow('ML Detection Debug', debug_frame)
                    
                    # Save calibration data
                    outfile = open(filename, 'wb')
                    pickle.dump([is_machine_learning, [pts1, side_view_compensation, rotation_count]], outfile)
                    outfile.close()
                    debug_log(f"ML-based calibration data saved: side_view={side_view_compensation}, rotation={rotation_count}")
                    
                    # Show info dialog if requested
                    if show_info:
                        if platform_name == "Darwin":
                            root = tk.Tk()
                            root.withdraw()
                        messagebox.showinfo(
                            "Chess Board Detected",
                            "Board detected successfully! Press any key to continue."
                        )
                        if platform_name == "Darwin":
                            root.destroy()
                    
                    # Show final frame and wait for key
                    cv2.imshow('Chess Board Detected', visualize_ml_detection(frame, result))
                    cv2.waitKey(0)
                    video_capture_thread.stop()
                    cv2.destroyAllWindows()
                    print("Chess board calibration completed successfully!")
                    sys.exit(0)
                else:
                    detection_info = "ML Detection: Failed - Check lighting and positioning"
            except Exception as e:
                debug_log(f"Error in ML detection: {e}")
                detection_info = f"ML Detection error: {str(e)[:50]}"
        else:
            # Try traditional OpenCV-based corner detection
            try:
                # Apply preprocessing for better detection
                gray, binary = enhance_frame_for_detection(frame)
                
                # Show preprocessed images in debug mode
                if toggle_debug_view:
                    cv2.imshow('Preprocessed Binary', binary)
                
                # Find chessboard corners
                debug_log("Attempting OpenCV findChessboardCorners...")
                retval, corners = cv2.findChessboardCorners(
                    binary, 
                    patternSize=board_dimensions,
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH | 
                          cv2.CALIB_CB_FAST_CHECK | 
                          cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                if not retval:
                    # Try again with gray image if binary fails
                    retval, corners = cv2.findChessboardCorners(
                        gray, 
                        patternSize=board_dimensions,
                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | 
                              cv2.CALIB_CB_FAST_CHECK | 
                              cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                
                if retval:
                    debug_log("Chessboard corners detected!")
                    detection_info = "OpenCV Detection: Success"
                    
                    # Draw the corners on a debug view
                    if toggle_debug_view:
                        corners_frame = frame.copy()
                        cv2.drawChessboardCorners(corners_frame, board_dimensions, corners, retval)
                        cv2.imshow('OpenCV Corner Detection', corners_frame)
                    
                    # Show info dialog if requested
                    if show_info:
                        if platform_name == "Darwin":
                            root = tk.Tk()
                            root.withdraw()
                        messagebox.showinfo(
                            "Chess Board Detected",
                            'Please check that corners of your chess board are correctly detected. '
                            'The square covered by points (0,0), (0,1),(1,0) and (1,1) should be a8. '
                            'You can rotate the image by pressing key "r" to adjust that. '
                            'Press key "q" to save detected chess board corners and finish board calibration.'
                        )
                        if platform_name == "Darwin":
                            root.destroy()
                    
                    # Process the corners
                    if corners[0][0][0] > corners[-1][0][0]:  # corners returned in reverse order
                        corners = corners[::-1]
                    
                    # Create augmented corners (9x9 grid)
                    debug_log("Creating augmented corners grid...")
                    augmented_corners = []
                    
                    # First row augmentation
                    row = []
                    for i in range(6):
                        corner1 = corners[i]
                        corner2 = corners[i + 8]
                        x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
                        y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
                        row.append((x, y))

                    for i in range(4, 7):
                        corner1 = corners[i]
                        corner2 = corners[i + 6]
                        x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
                        y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
                        row.append((x, y))
                    
                    augmented_corners.append(row)
                    
                    # Middle rows augmentation
                    for i in range(7):
                        row = []
                        corner1 = corners[i * 7]
                        corner2 = corners[i * 7 + 1]
                        x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
                        y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
                        row.append((x, y))

                        for corner in corners[i * 7:(i + 1) * 7]:
                            x = corner[0][0]
                            y = corner[0][1]
                            row.append((x, y))

                        corner1 = corners[i * 7 + 6]
                        corner2 = corners[i * 7 + 5]
                        x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
                        y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
                        row.append((x, y))
                        augmented_corners.append(row)
                    
                    # Last row augmentation
                    row = []
                    for i in range(6):
                        corner1 = corners[42 + i]
                        corner2 = corners[42 + i - 6]
                        x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
                        y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
                        row.append((x, y))

                    for i in range(4, 7):
                        corner1 = corners[42 + i]
                        corner2 = corners[42 + i - 8]
                        x = corner1[0][0] + (corner1[0][0] - corner2[0][0])
                        y = corner1[0][1] + (corner1[0][1] - corner2[0][1])
                        row.append((x, y))
                    
                    augmented_corners.append(row)
                    
                    # Ensure a8 is in top-left corner
                    while augmented_corners[0][0][0] > augmented_corners[8][8][0] or augmented_corners[0][0][1] > \
                            augmented_corners[8][8][1]:
                        rotateMatrix(augmented_corners)
                    
                    # Create perspective transform points
                    pts1 = np.float32([list(augmented_corners[0][0]), list(augmented_corners[8][0]), 
                                      list(augmented_corners[0][8]), list(augmented_corners[8][8])])
                    
                    # Process empty board to create ROI mask
                    empty_board = perspective_transform(frame, pts1)
                    edges = edge_detection(empty_board)
                    kernel = np.ones((7, 7), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)
                    roi_mask = cv2.bitwise_not(edges)
                    
                    # Clean up mask edges
                    roi_mask[:7, :] = 0
                    roi_mask[:, :7] = 0
                    roi_mask[-7:, :] = 0
                    roi_mask[:, -7:] = 0
                    
                    # Let user adjust rotation if needed
                    rotation_count = 0
                    while True:
                        marked_frame = mark_corners(frame.copy(), augmented_corners, rotation_count)
                        display_frame = draw_calibration_overlay(marked_frame, found_board=True, 
                                                               fps=fps_value, detection_info=detection_info)
                        cv2.imshow('Chess Board Calibration', display_frame)
                        
                        # Wait for user input
                        response = cv2.waitKey(0)
                        if response & 0xFF == ord('r'):
                            rotation_count += 1
                            rotation_count %= 4
                            debug_log(f"Rotated board (new rotation count: {rotation_count})")
                        elif response & 0xFF == ord('q'):
                            debug_log("Board position accepted")
                            break
                    
                    # Calculate side view compensation
                    first_row = euclidean_distance(augmented_corners[1][1], augmented_corners[1][7])
                    last_row = euclidean_distance(augmented_corners[7][1], augmented_corners[7][7])
                    first_column = euclidean_distance(augmented_corners[1][1], augmented_corners[7][1])
                    last_column = euclidean_distance(augmented_corners[1][7], augmented_corners[7][7])

                    if abs(first_row - last_row) >= abs(first_column - last_column):
                        if first_row >= last_row:
                            side_view_compensation = (1, 0)
                        else:
                            side_view_compensation = (-1, 0)
                    else:
                        if first_column >= last_column:
                            side_view_compensation = (0, -1)
                        else:
                            side_view_compensation = (0, 1)

                    debug_log("Side view compensation: " + str(side_view_compensation))
                    debug_log("Rotation count: " + str(rotation_count))
                    
                    # Save calibration data
                    outfile = open(filename, 'wb')
                    pickle.dump([is_machine_learning, [augmented_corners, side_view_compensation, rotation_count, roi_mask]], outfile)
                    outfile.close()
                    
                    debug_log("Chess board calibration completed successfully!")
                    break
                else:
                    detection_info = "OpenCV Detection: Failed - Check lighting and alignment"
            except Exception as e:
                debug_log(f"Error in OpenCV detection: {e}")
                detection_info = f"OpenCV Detection error: {str(e)[:50]}"
        
        # Display the frame with overlay
        display_frame = draw_calibration_overlay(frame, found_board=False, 
                                               fps=fps_value, detection_info=detection_info)
        cv2.imshow('Chess Board Calibration', display_frame)
        
        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            debug_log("Calibration cancelled by user")
            break
        elif key == ord('d'):
            toggle_debug_view = not toggle_debug_view
            debug_log(f"Debug view {'enabled' if toggle_debug_view else 'disabled'}")
            if not toggle_debug_view:
                # Close debug windows when toggling off
                cv2.destroyWindow('Debug Processing')
                cv2.destroyWindow('Preprocessed Binary')
                cv2.destroyWindow('OpenCV Corner Detection')
                cv2.destroyWindow('ML Detection Debug')
        
        # Calculate loop time for performance monitoring
        loop_time = time.time() - loop_start
        if loop_time > 0.1:  # Log slow frames
            debug_log(f"Slow frame processing: {loop_time:.3f}s")
    
    # Clean up
    debug_log("Shutting down...")
    video_capture_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
