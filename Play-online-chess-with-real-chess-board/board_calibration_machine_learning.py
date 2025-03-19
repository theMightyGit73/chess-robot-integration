import numpy as np
import cv2
import logging
import time
from typing import Tuple, List, Optional, Dict, Any
from helper import euclidean_distance, perspective_transform, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BoardCalibration")

def detect_board(
    original_image: np.ndarray, 
    corner_model: cv2.dnn.Net, 
    piece_model: cv2.dnn.Net, 
    color_model: cv2.dnn.Net
) -> Optional[Tuple[np.ndarray, Tuple[int, int], int]]:
    """
    Detect and calibrate a chess board in an image using machine learning models.
    
    Args:
        original_image: Input image containing a chess board
        corner_model: YOLO model for detecting board corners
        piece_model: CNN model for detecting chess pieces
        color_model: CNN model for determining piece colors
        
    Returns:
        Tuple containing (perspective transform points, side view compensation, rotation count)
        or None if board detection fails
        
    Raises:
        ValueError: If image or models are invalid
    """
    try:
        # Validate inputs
        if original_image is None or original_image.size == 0:
            raise ValueError("Invalid input image")
            
        if corner_model is None or piece_model is None or color_model is None:
            raise ValueError("Invalid model(s)")
            
        # Start timing for performance tracking
        start_time = time.time()
        logger.info("Starting chess board detection")
            
        # Get image dimensions
        try:
            [height, width, channels] = original_image.shape
            if channels != 3:
                logger.warning(f"Expected 3-channel image, got {channels} channels")
        except ValueError:
            raise ValueError("Input image must be a color image with 3 channels")
            
        # Pad image to square for consistent scaling
        length = max(height, width)
        padded_image = np.zeros((length, length, 3), np.uint8)
        padded_image[0:height, 0:width] = original_image
        
        # Calculate scale factor for later coordinate conversion
        scale = length / 640
        logger.debug(f"Image dimensions: {width}x{height}, scale factor: {scale}")
        
        # Prepare image for YOLO model
        try:
            blob = cv2.dnn.blobFromImage(
                padded_image, 
                scalefactor=1/255,
                size=(640, 640),
                swapRB=True
            )
            
            # Set input and run corner detection model
            corner_model.setInput(blob)
            outputs = corner_model.forward()
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]
            
            logger.debug(f"Corner detection network output shape: {outputs.shape}")
            
        except Exception as e:
            logger.error(f"Error running corner detection model: {e}")
            return None
        
        # Process model outputs to get corner boxes
        boxes = []
        scores = []
        class_ids = []
        
        try:
            # Extract detection data
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                
                # Filter by confidence threshold
                if maxScore >= 0.25:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y
                        outputs[0][i][2],  # width
                        outputs[0][i][3]   # height
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)
                    
            logger.debug(f"Found {len(boxes)} potential corners")
                    
            # Apply non-maximum suppression to eliminate overlapping detections
            if len(boxes) > 0:
                result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
            else:
                logger.warning("No corner boxes detected")
                return None
            
        except Exception as e:
            logger.error(f"Error processing detection outputs: {e}")
            return None
        
        # Process detections to get corner points
        try:
            detections = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    'confidence': scores[index],
                    'box': box,
                }
                detections.append(detection)
            
            # Ensure we have enough corners for a quadrilateral
            if len(detections) < 4:
                logger.warning(f"Insufficient corners detected: found {len(detections)}, need 4")
                return None
                
            # Sort by confidence and take top 4
            detections.sort(key=lambda detection: detection['confidence'], reverse=True)
            detections = detections[:4]
            
            # Calculate center points of detected corner boxes
            middle_points = []
            for detection in detections:
                box = detection['box']
                x, y, w, h = box
                middle_x = (x + (w / 2)) * scale
                middle_y = (y + (h / 2)) * scale
                middle_points.append([middle_x, middle_y])
                
            logger.debug(f"Corner middle points: {middle_points}")
            
        except Exception as e:
            logger.error(f"Error processing corner detections: {e}")
            return None
        
        # Find the corner positions (top-left, top-right, bottom-left, bottom-right)
        try:
            # Find bounding rectangle dimensions
            minX = min(point[0] for point in middle_points)
            minY = min(point[1] for point in middle_points)
            maxX = max(point[0] for point in middle_points)
            maxY = max(point[1] for point in middle_points)
            
            # Assign corner positions based on distances to extremes
            top_left = min(middle_points, key=lambda point: euclidean_distance(point, [minX, minY]))
            top_right = min(middle_points, key=lambda point: euclidean_distance(point, [maxX, minY]))
            bottom_left = min(middle_points, key=lambda point: euclidean_distance(point, [minX, maxY]))
            bottom_right = min(middle_points, key=lambda point: euclidean_distance(point, [maxX, maxY]))
            
            logger.debug("Corner positions identified")
            
        except Exception as e:
            logger.error(f"Error determining corner positions: {e}")
            return None
        
        # Calculate side view compensation based on board geometry
        try:
            # Measure the edges of the board
            first_row = euclidean_distance(top_left, top_right)
            last_row = euclidean_distance(bottom_left, bottom_right)
            first_column = euclidean_distance(top_left, bottom_left)
            last_column = euclidean_distance(top_right, bottom_right)
            
            # Determine if horizontal or vertical compensation is needed
            if abs(first_row - last_row) >= abs(first_column - last_column):
                # Horizontal compensation
                if first_row >= last_row:
                    side_view_compensation = (1, 0)  # Bottom edge is closer to camera
                else:
                    side_view_compensation = (-1, 0)  # Top edge is closer to camera
            else:
                # Vertical compensation
                if first_column >= last_column:
                    side_view_compensation = (0, -1)  # Right edge is closer to camera
                else:
                    side_view_compensation = (0, 1)  # Left edge is closer to camera
                    
            logger.debug(f"Side view compensation: {side_view_compensation}")
            
        except Exception as e:
            logger.error(f"Error calculating side view compensation: {e}")
            # Use default if calculation fails
            side_view_compensation = (0, 0)
        
        # Create perspective transform matrix
        pts1 = np.float32([top_left, bottom_left, top_right, bottom_right])
        
        # Transform the image to get a top-down view
        try:
            board_image = perspective_transform(original_image, pts1)
            logger.debug("Perspective transform applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying perspective transform: {e}")
            return None
        
        # Determine board rotation by detecting black pieces along different edges
        try:
            # Define the four possible orientations to check
            squares_to_check_for_rotation_count = [
                [(0, i) for i in range(7)],  # Top row
                [(i, 0) for i in range(7)],  # Left column
                [(7, i) for i in range(7)],  # Bottom row
                [(i, 7) for i in range(7)],  # Right column
            ]
            
            # Find which orientation has the most black pieces
            rotation_count = 0
            max_score = 0
            
            for i in range(len(squares_to_check_for_rotation_count)):
                current_score = 0
                
                # Check each square in this orientation
                for row, column in squares_to_check_for_rotation_count[i]:
                    # Get square dimensions
                    height, width = board_image.shape[:2]
                    minX = int(column * width / 8)
                    maxX = int((column + 1) * width / 8)
                    minY = int(row * height / 8)
                    maxY = int((row + 1) * height / 8)
                    
                    # Extract the square image
                    try:
                        square_image = board_image[minY:maxY, minX:maxX]
                        
                        # Check if there's a piece and if it's black
                        is_piece = predict(square_image, piece_model)
                        if is_piece:
                            is_white = predict(square_image, color_model)
                            if not is_white:
                                current_score += 1
                    except Exception as square_error:
                        logger.warning(f"Error processing square ({row}, {column}): {square_error}")
                        continue
                
                # Update best rotation if this is better
                if current_score > max_score:
                    max_score = current_score
                    rotation_count = i
                    
            logger.debug(f"Detected rotation: {rotation_count} (score: {max_score})")
            
        except Exception as e:
            logger.error(f"Error determining board rotation: {e}")
            rotation_count = 0  # Default to no rotation
        
        # Visualize the detected board (for debugging)
        try:
            # Define colors for visualization
            green_color = (0, 255, 0)
            blue_color = (255, 0, 0)
            red_color = (0, 0, 255)
            
            # Convert points to integer coordinates
            top_left_int = (int(top_left[0]), int(top_left[1]))
            top_right_int = (int(top_right[0]), int(top_right[1]))
            bottom_left_int = (int(bottom_left[0]), int(bottom_left[1]))
            bottom_right_int = (int(bottom_right[0]), int(bottom_right[1]))
            
            # Draw lines with colors indicating orientation
            if rotation_count == 0:
                cv2.line(original_image, top_left_int, top_right_int, green_color, 5)  # Top (green)
                cv2.line(original_image, top_right_int, bottom_right_int, red_color, 5)  # Right (red)
                cv2.line(original_image, bottom_left_int, bottom_right_int, blue_color, 5)  # Bottom (blue)
                cv2.line(original_image, top_left_int, bottom_left_int, red_color, 5)  # Left (red)
            elif rotation_count == 1:
                cv2.line(original_image, top_left_int, top_right_int, red_color, 5)
                cv2.line(original_image, top_right_int, bottom_right_int, blue_color, 5)
                cv2.line(original_image, bottom_left_int, bottom_right_int, red_color, 5)
                cv2.line(original_image, top_left_int, bottom_left_int, green_color, 5)
            elif rotation_count == 2:
                cv2.line(original_image, top_left_int, top_right_int, blue_color, 5)
                cv2.line(original_image, top_right_int, bottom_right_int, red_color, 5)
                cv2.line(original_image, bottom_left_int, bottom_right_int, green_color, 5)
                cv2.line(original_image, top_left_int, bottom_left_int, red_color, 5)
            elif rotation_count == 3:
                cv2.line(original_image, top_left_int, top_right_int, red_color, 5)
                cv2.line(original_image, top_right_int, bottom_right_int, green_color, 5)
                cv2.line(original_image, bottom_left_int, bottom_right_int, red_color, 5)
                cv2.line(original_image, top_left_int, bottom_left_int, blue_color, 5)
                
        except Exception as e:
            logger.warning(f"Error drawing visualization: {e}")
        
        # Log the results
        logger.info(f"Side view compensation: {side_view_compensation}")
        logger.info(f"Rotation count: {rotation_count}")
        
        # Calculate and log elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Board detection completed in {elapsed_time:.2f} seconds")
        
        # Return the detected parameters
        return pts1, side_view_compensation, rotation_count
        
    except Exception as e:
        logger.error(f"Unhandled error in detect_board: {e}")
        return None


def test_board_detection(image_path: str, model_paths: Dict[str, str]) -> bool:
    """
    Test the board detection functionality on a sample image.
    
    Args:
        image_path: Path to test image
        model_paths: Dictionary with paths to the required models
        
    Returns:
        True if detection succeeds, False otherwise
    """
    try:
        # Load test image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image from {image_path}")
            return False
            
        # Load models
        corner_model = cv2.dnn.readNetFromONNX(model_paths["corner"])
        piece_model = cv2.dnn.readNetFromONNX(model_paths["piece"])
        color_model = cv2.dnn.readNetFromONNX(model_paths["color"])
        
        # Run detection
        print("Running board detection test...")
        result = detect_board(image, corner_model, piece_model, color_model)
        
        if result:
            pts1, side_view_compensation, rotation_count = result
            print(f"Detection successful!")
            print(f"Side view compensation: {side_view_compensation}")
            print(f"Rotation count: {rotation_count}")
            
            # Save visualization
            output_path = "board_detection_result.jpg"
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to {output_path}")
            
            # Also save the perspective-transformed image
            warped = perspective_transform(image, pts1)
            cv2.imwrite("board_perspective.jpg", warped)
            
            return True
        else:
            print("Board detection failed")
            return False
            
    except Exception as e:
        print(f"Test error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use provided image path
        image_path = sys.argv[1]
    else:
        # Use default (assuming camera capture)
        import cv2
        
        # Try to capture from camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
            
        # Wait for camera to stabilize
        for _ in range(10):
            ret, _ = camera.read()
            time.sleep(0.1)
            
        # Capture image
        ret, image = camera.read()
        camera.release()
        
        if not ret:
            print("Error: Could not capture image from camera")
            sys.exit(1)
            
        # Save captured image
        image_path = "captured_board.jpg"
        cv2.imwrite(image_path, image)
        print(f"Captured image saved to {image_path}")
    
    # Define model paths
    model_paths = {
        "corner": "yolo_corner.onnx",
        "piece": "cnn_piece.onnx",
        "color": "cnn_color.onnx"
    }
    
    # Run the test
    success = test_board_detection(image_path, model_paths)
    
    if not success:
        print("Board detection test failed")
        sys.exit(1)
