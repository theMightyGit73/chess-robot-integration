import cv2
import numpy as np
import logging
from math import sqrt
import time
import functools
import traceback
import os

# Configure logging with rotating file handler
log_dir = os.path.join(os.path.expanduser("~"), ".chess_robot", "logs")
os.makedirs(log_dir, exist_ok=True)

# Setup logger
logger = logging.getLogger("ChessHelpers")
logger.setLevel(logging.INFO)  # Default level, can be changed at runtime

# Create handlers and add to logger
try:
    from logging.handlers import RotatingFileHandler
    log_file = os.path.join(log_dir, "helper.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
except Exception as e:
    print(f"Warning: Error setting up logging: {e}")

# Global settings
DEBUG_MODE = False  # When True, additional debug logs and visualizations will be enabled
CACHE_ENABLED = True  # Can be disabled for debugging
_cache = {}  # Simple cache for function results

def set_debug_mode(enabled=False):
    """Enable or disable debug mode globally."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)
    logger.debug(f"Debug mode {'enabled' if enabled else 'disabled'}")

def clear_cache():
    """Clear all cached results."""
    global _cache
    cache_size = len(_cache)
    _cache.clear()
    logger.debug(f"Cache cleared ({cache_size} items)")

def error_handler(func):
    """Decorator for handling exceptions in functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            if DEBUG_MODE:
                logger.debug(traceback.format_exc())
            
            # Return appropriate default values based on function type
            if func.__name__ == 'perspective_transform':
                # For transform function, return a blank image of expected size
                return np.zeros((480, 480, 3), dtype=np.uint8)
            elif func.__name__ == 'euclidean_distance':
                return 0.0
            elif func.__name__ == 'predict':
                if kwargs.get('return_confidence', False):
                    return 0.0
                return 0
            elif func.__name__ in ['edge_detection', 'auto_canny']:
                return np.zeros((1, 1), dtype=np.uint8)
            elif func.__name__ == 'detect_state':
                return [[[] for _ in range(8)] for _ in range(8)]
            else:
                # For other functions, reraise the exception
                raise
    return wrapper

def timed(func):
    """Decorator for timing function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_MODE:
            return func(*args, **kwargs)
            
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Log if execution time is longer than expected
        if execution_time > 100:  # Log operations taking more than 100ms
            logger.debug(f"{func.__name__} took {execution_time:.2f}ms (slow)")
        else:
            logger.debug(f"{func.__name__} took {execution_time:.2f}ms")
            
        return result
    return wrapper

def cached(func):
    """Decorator for caching function results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not CACHE_ENABLED:
            return func(*args, **kwargs)
            
        # Create a cache key from the function name and arguments
        try:
            # For numpy arrays, use their shape and a hash of sample data
            processed_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # Use first pixels and shape as part of key
                    if arg.size > 0:
                        # Take a small sample to speed up hashing
                        sample_size = min(1000, arg.size)
                        flat_data = arg.flatten()[:sample_size]
                        arg_hash = hash(bytes(flat_data))
                        processed_args.append((arg.shape, arg_hash))
                    else:
                        processed_args.append((arg.shape, 0))
                else:
                    processed_args.append(arg)
                    
            # Process kwargs similarly
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    if v.size > 0:
                        sample_size = min(1000, v.size)
                        flat_data = v.flatten()[:sample_size]
                        v_hash = hash(bytes(flat_data))
                        processed_kwargs[k] = (v.shape, v_hash)
                    else:
                        processed_kwargs[k] = (v.shape, 0)
                else:
                    processed_kwargs[k] = v
            
            # Create the cache key
            cache_key = (func.__name__, tuple(processed_args), frozenset(processed_kwargs.items()))
            
            if cache_key in _cache:
                if DEBUG_MODE:
                    logger.debug(f"Cache hit for {func.__name__}")
                return _cache[cache_key]
                
            result = func(*args, **kwargs)
            _cache[cache_key] = result
            
            # Limit cache size to prevent memory issues
            if len(_cache) > 1000:
                # Remove oldest item (in practice, we could use LRU cache)
                _cache.pop(next(iter(_cache.keys())))
                
            return result
        except (TypeError, ValueError):
            # If we can't create a proper cache key, just call the function
            logger.debug(f"Cache key creation failed for {func.__name__}")
            return func(*args, **kwargs)
    return wrapper

@error_handler
@timed
def euclidean_distance(first, second):
    """
    Compute the Euclidean distance between two 2D points.
    
    Args:
        first: A tuple representing (x, y) coordinates of first point
        second: A tuple representing (x, y) coordinates of second point
        
    Returns:
        The Euclidean distance as a float
    """
    # Validate inputs
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        logger.warning(f"Invalid points: {first}, {second}")
        return 0.0
        
    if len(first) < 2 or len(second) < 2:
        logger.warning(f"Points missing coordinates: {first}, {second}")
        return 0.0
    
    # Fast computation
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    distance = sqrt(dx*dx + dy*dy)
    
    return distance

@error_handler
@timed
@cached
def perspective_transform(image, pts1, dimension=480):
    """
    Perform a perspective transform on an image.
    
    Args:
        image: Input image
        pts1: Source points (4x2 numpy array)
        dimension: Size of the output square image
        
    Returns:
        The warped (transformed) image
    """
    # Validate inputs
    if image is None or image.size == 0:
        logger.warning("Invalid input image for perspective_transform")
        return np.zeros((dimension, dimension, 3), dtype=np.uint8)
        
    if pts1 is None or len(pts1) != 4:
        logger.warning(f"Invalid points array: {pts1}")
        return np.zeros((dimension, dimension, 3), dtype=np.uint8)
    
    # Ensure pts1 has correct shape and type
    try:
        pts1 = np.float32(pts1)
    except Exception as e:
        logger.error(f"Error converting pts1 to float32: {e}")
        return np.zeros((dimension, dimension, 3), dtype=np.uint8)
    
    # Define destination points for a square output
    pts2 = np.float32([[0, 0], [0, dimension], [dimension, 0], [dimension, dimension]])
    
    try:
        # Calculate transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Apply the transformation
        dst = cv2.warpPerspective(image, M, (dimension, dimension))
        
        return dst
    except Exception as e:
        logger.error(f"Error in perspective transform: {e}")
        return np.zeros((dimension, dimension, 3), dtype=np.uint8)

@error_handler
def rotateMatrix(matrix):
    """
    Rotate a square matrix in-place by 90 degrees clockwise.
    
    Args:
        matrix: A square matrix (list of lists) to rotate
    """
    if not matrix:
        logger.warning("Empty matrix provided to rotateMatrix")
        return
        
    # Validate matrix is square
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            logger.warning("Non-square matrix provided to rotateMatrix")
            return
    
    # More efficient rotation algorithm using O(1) extra space
    for layer in range(size // 2):
        first = layer
        last = size - 1 - layer
        
        for i in range(first, last):
            offset = i - first
            
            # Save top
            top = matrix[first][i]
            
            # Left -> Top
            matrix[first][i] = matrix[last - offset][first]
            
            # Bottom -> Left
            matrix[last - offset][first] = matrix[last][last - offset]
            
            # Right -> Bottom
            matrix[last][last - offset] = matrix[i][last]
            
            # Top -> Right
            matrix[i][last] = top
    
    if DEBUG_MODE:
        logger.debug(f"Matrix of size {size}x{size} rotated clockwise")

@error_handler
@timed
@cached
def auto_canny(image, sigma_upper=0.2, sigma_lower=0.8):
    """
    Apply automatic Canny edge detection using the median of the pixel intensities.
    
    Args:
        image: Grayscale image
        sigma_upper: Upper sigma multiplier
        sigma_lower: Lower sigma multiplier
        
    Returns:
        The edge-detected image
    """
    # Validate input
    if image is None or image.size == 0:
        logger.warning("Invalid input to auto_canny")
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Ensure image is grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    try:
        # Calculate the median pixel intensity
        median_intensity = np.median(image)
        
        # Apply automatic thresholding
        lower = int(max(0, (1.0 - sigma_lower) * median_intensity))
        upper = int(min(255, (1.0 + sigma_upper) * median_intensity))
        
        # Apply Canny edge detection
        edged = cv2.Canny(image, lower, upper)
        
        return edged
    except Exception as e:
        logger.error(f"Error in auto_canny: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)

@error_handler
@timed
@cached
def edge_detection(frame):
    """
    Perform edge detection on a multi-channel image.
    
    Args:
        frame: Input color image
        
    Returns:
        The edge-detected image
    """
    # Validate input
    if frame is None or frame.size == 0:
        logger.warning("Invalid input to edge_detection")
        return np.zeros((1, 1), dtype=np.uint8)
    
    try:
        # Check if image is already grayscale
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            # For grayscale images, just apply edge detection directly
            gray = frame if len(frame.shape) == 2 else frame[:,:,0]
            return auto_canny(gray)
        
        # Create CLAHE object for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Create morphological kernel
        kernel = np.ones((3, 3), np.uint8)
        
        # Process each channel separately for better edge detection
        channels = cv2.split(frame)
        edges = []
        
        for idx, gray in enumerate(channels):
            # Apply morphological closing to reduce noise
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Enhance contrast
            gray = clahe.apply(gray)
            
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply auto Canny for edge detection
            edge = auto_canny(gray)
            edges.append(edge)
        
        # Combine edges from all channels
        combined = cv2.bitwise_or(cv2.bitwise_or(edges[0], edges[1]), edges[2])
        
        # Apply morphological closing to connect nearby edges
        kernel2 = np.ones((3, 3), np.uint8)
        final_edges = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel2)
        
        return final_edges
    except Exception as e:
        logger.error(f"Error in edge_detection: {e}")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

@error_handler
@timed
@cached
def get_square_image(row, column, board_img):
    """
    Extract a square from the chess board image.
    
    Args:
        row: Row index (0-7)
        column: Column index (0-7)
        board_img: The chess board image
        
    Returns:
        The extracted square image
    """
    # Validate inputs
    if board_img is None or board_img.size == 0:
        logger.warning("Invalid board image in get_square_image")
        return np.zeros((1, 1), dtype=np.uint8)
        
    if not (0 <= row <= 7 and 0 <= column <= 7):
        logger.warning(f"Invalid chess square coordinates: ({row}, {column})")
        return np.zeros((1, 1), dtype=np.uint8)
    
    try:
        # Get image dimensions
        height, width = board_img.shape[:2]
        
        # Calculate square boundaries
        minX = int(column * width / 8)
        maxX = int((column + 1) * width / 8)
        minY = int(row * height / 8)
        maxY = int((row + 1) * height / 8)
        
        # Extract the square
        square = board_img[minY:maxY, minX:maxX]
        
        # Remove borders if square is large enough
        border_size = 3
        if square.shape[0] > 2 * border_size and square.shape[1] > 2 * border_size:
            square_without_borders = square[border_size:-border_size, border_size:-border_size]
            return square_without_borders
        else:
            return square
    except Exception as e:
        logger.error(f"Error extracting square ({row}, {column}): {e}")
        return np.zeros((1, 1), dtype=np.uint8)

@error_handler
@timed
def contains_piece(square, view):
    """
    Determine if a chess piece is contained within a square.
    
    Args:
        square: Image of the square
        view: Tuple indicating which half of the square to inspect
              (0,-1): right half, (0,1): left half, 
              (1,0): bottom half, (-1,0): top half
              
    Returns:
        List indicating detection: [True] for piece, [True, False] for uncertain,
        [False] for empty
    """
    # Validate inputs
    if square is None or square.size == 0:
        logger.warning("Invalid square image in contains_piece")
        return [False]
    
    try:
        # Get square dimensions
        height, width = square.shape[:2]
        
        # Extract the appropriate half based on view parameter
        if view == (0, -1):
            half = square[:, width // 2:]
        elif view == (0, 1):
            half = square[:, :width // 2]
        elif view == (1, 0):
            half = square[height // 2:, :]
        elif view == (-1, 0):
            half = square[:height // 2, :]
        else:
            logger.warning(f"Invalid view parameter: {view}")
            half = square  # Default to full square
        
        # Calculate means for classification
        half_mean = float(np.mean(half))
        square_mean = float(np.mean(square))
        
        # Apply classification logic
        if half_mean < 1.0:
            return [False]  # Definitely empty
        elif square_mean > 15.0:
            return [True]   # Definitely has piece
        elif square_mean > 6.0:
            return [True, False]  # Uncertain, but likely has piece
        else:
            if square_mean > 2.0 and DEBUG_MODE:
                logger.debug(f"Square mean low but above threshold: {square_mean:.2f}")
            return [False]  # Likely empty
    except Exception as e:
        logger.error(f"Error in contains_piece: {e}")
        return [False]

@error_handler
@timed
def detect_state(frame, view, roi_mask):
    """
    Detect the state of a chessboard.
    
    Args:
        frame: Image of the chessboard
        view: Parameter for contains_piece
        roi_mask: Region of interest mask
        
    Returns:
        A 2D list representing the board state
    """
    # Validate inputs
    if frame is None or frame.size == 0:
        logger.warning("Invalid frame in detect_state")
        return [[[] for _ in range(8)] for _ in range(8)]
    
    if roi_mask is None or roi_mask.size == 0:
        logger.warning("Invalid ROI mask in detect_state")
        # Create a default mask if none provided
        roi_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    try:
        # Detect edges in the frame
        edges = edge_detection(frame)
        
        # Apply the ROI mask
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Extract squares and check for pieces
        board_image = []
        for row in range(8):
            row_squares = []
            for column in range(8):
                square = get_square_image(row, column, masked_edges)
                row_squares.append(square)
            board_image.append(row_squares)
        
        # Determine pieces for each square
        result = []
        for row in range(8):
            row_result = []
            for column in range(8):
                piece_result = contains_piece(board_image[row][column], view)
                row_result.append(piece_result)
            result.append(row_result)
        
        # For debugging: count detected pieces
        if DEBUG_MODE:
            piece_count = sum(1 for row in result for square in row if True in square)
            logger.debug(f"Detected {piece_count} potential pieces on the board")
        
        return result
    except Exception as e:
        logger.error(f"Error in detect_state: {e}")
        return [[[] for _ in range(8)] for _ in range(8)]

@error_handler
@timed
@cached
def predict(image, model, return_confidence=False):
    """
    Predict if an image contains a chess piece using a neural network model.
    
    Args:
        image: Input image of a chess square
        model: Neural network model
        return_confidence: If True, returns confidence score instead of label
        
    Returns:
        If return_confidence is True: confidence score (float)
        If return_confidence is False: class label (int)
    """
    # Validate inputs
    if image is None or image.size == 0:
        logger.warning("Invalid image in predict")
        return 0.0 if return_confidence else 0
    
    if model is None:
        logger.warning("Invalid model in predict")
        return 0.0 if return_confidence else 0
    
    try:
        # Resize the image to the expected input size
        processed = cv2.resize(image, (64, 64))
        
        # Normalize the image
        processed = processed.astype(np.float32) / 255.0
        
        # Transpose to channel-first format (NCHW)
        processed = np.transpose(processed, (2, 0, 1))
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        # Set the model input
        model.setInput(processed)
        
        # Perform forward pass
        output = model.forward()
        
        # Get the result
        if return_confidence:
            confidence = float(np.max(output))
            return confidence
        else:
            label = int(np.argmax(output))
            return label
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return 0.0 if return_confidence else 0

def get_square_name(row, column, rotation_count=0):
    """
    Convert row and column to chess square name (a1, b2, etc.),
    accounting for board rotation.
    
    Args:
        row: Row index (0-7)
        column: Column index (0-7)
        rotation_count: Number of 90-degree clockwise rotations (0-3)
        
    Returns:
        Chess square name (e.g. 'e4')
    """
    # Validate inputs
    if not (0 <= row <= 7 and 0 <= column <= 7):
        logger.warning(f"Invalid chess coordinates: ({row}, {column})")
        return "a1"  # Default to a1 for invalid coordinates
    
    # Apply rotation transformation
    if rotation_count % 4 == 0:
        adj_row, adj_col = row, column
    elif rotation_count % 4 == 1:
        adj_row, adj_col = column, 7-row
    elif rotation_count % 4 == 2:
        adj_row, adj_col = 7-row, 7-column
    elif rotation_count % 4 == 3:
        adj_row, adj_col = 7-column, row
    
    # Convert to chess notation
    file_letter = chr(ord('a') + adj_col)
    rank_number = 8 - adj_row
    
    return f"{file_letter}{rank_number}"

def draw_chessboard_grid(frame, highlight_squares=None):
    """
    Draw a chessboard grid overlay on an image.
    
    Args:
        frame: The original frame
        highlight_squares: Optional list of squares to highlight with format [(row, col, color)]
        
    Returns:
        An image with chessboard grid overlay
    """
    if frame is None or frame.size == 0:
        logger.warning("Invalid frame in draw_chessboard_grid")
        return frame
    
    result = frame.copy()
    height, width = result.shape[:2]
    
    # Draw horizontal and vertical lines
    for i in range(9):
        y = i * height // 8
        x = i * width // 8
        cv2.line(result, (0, y), (width, y), (128, 128, 128), 1)
        cv2.line(result, (x, 0), (x, height), (128, 128, 128), 1)
    
    # Highlight specific squares if requested
    if highlight_squares:
        for row, col, color in highlight_squares:
            if 0 <= row < 8 and 0 <= col < 8:
                minX = int(col * width / 8)
                maxX = int((col + 1) * width / 8)
                minY = int(row * height / 8)
                maxY = int((row + 1) * height / 8)
                
                # Draw rectangle around the square
                cv2.rectangle(result, (minX, minY), (maxX, maxY), color, 2)
    
    # Add coordinate labels (a-h, 1-8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_color = (200, 200, 200)
    thickness = 1
    
    # Add column labels (a-h)
    for i in range(8):
        x = i * width // 8 + width // 16
        y = height - 5
        cv2.putText(result, chr(97 + i), (x, y), font, font_scale, text_color, thickness)
    
    # Add row labels (1-8)
    for i in range(8):
        x = 5
        y = i * height // 8 + height // 16
        cv2.putText(result, str(8 - i), (x, y), font, font_scale, text_color, thickness)
    
    return result
