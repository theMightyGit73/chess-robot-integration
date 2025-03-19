import sys
import os
import numpy as np
import cv2
import time
from statistics import median
import subprocess

class Board_position:
    """Class to store the position of the detected chess board on screen"""
    def __init__(self, minX, minY, maxX, maxY):
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

def take_screenshot():
    """Take a screenshot using a method that works on Raspberry Pi"""
    try:
        # Try using scrot (a lightweight screenshot tool for Linux)
        screenshot_path = '/tmp/chess_screenshot.png'
        subprocess.run(['scrot', screenshot_path], check=True)
        screenshot = cv2.imread(screenshot_path)
        os.remove(screenshot_path)  # Clean up
        
        if screenshot is None:
            raise Exception("Failed to capture screenshot with scrot")
            
        return screenshot
        
    except Exception as e:
        print(f"Error taking screenshot with scrot: {e}")
        
        try:
            # Fallback to mss
            import mss
            import mss.tools
            
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = np.array(sct.grab(monitor))
                # Convert from BGRA to BGR
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                return screenshot
                
        except Exception as e2:
            print(f"Error taking screenshot with mss: {e2}")
            
            try:
                # Final fallback to pyautogui
                import pyautogui
                screenshot = np.array(pyautogui.screenshot())
                # Convert from RGB to BGR (OpenCV format)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
                
            except Exception as e3:
                print(f"Error taking screenshot with pyautogui: {e3}")
                print("No screenshot method available. Make sure at least one of these is installed:")
                print("  - scrot (install with: sudo apt install scrot)")
                print("  - mss (install with: pip install mss)")
                print("  - pyautogui (install with: pip install pyautogui)")
                return None

def find_chessboard():
    """Find the chess board on screen using template matching"""
    print("Taking screenshot for chess board detection...")
    large_image = take_screenshot()
    
    if large_image is None:
        print("Failed to take screenshot")
        return None, False
    
    # Check if template images exist
    if not os.path.exists("white.JPG"):
        print("Error: white.JPG template file not found")
        return None, False
        
    if not os.path.exists("black.JPG"):
        print("Error: black.JPG template file not found")
        return None, False
    
    print("Loading template images...")
    white_image = cv2.imread("white.JPG")
    black_image = cv2.imread("black.JPG")
    
    if white_image is None or black_image is None:
        print("Error loading template images")
        return None, False
    
    print("Performing template matching...")
    method = cv2.TM_SQDIFF_NORMED
    
    # Match both white and black templates
    result_white = cv2.matchTemplate(large_image, white_image, method)
    result_black = cv2.matchTemplate(large_image, black_image, method)
    
    # Determine which template matches better
    white_min_val = cv2.minMaxLoc(result_white)[0]
    black_min_val = cv2.minMaxLoc(result_black)[0]
    
    we_are_white = True
    result = result_white
    small_image = white_image
    
    if black_min_val < white_min_val:  # If black matches better
        result = result_black
        we_are_white = False
        small_image = black_image
        print("Detected: We are playing black")
    else:
        print("Detected: We are playing white")
    
    # Get the location of the match
    minimum_value, maximum_value, minimum_location, maximum_location = cv2.minMaxLoc(result)
    print(f"Match quality: {minimum_value:.4f} (lower is better)")
    
    if minimum_value > 0.2:
        print("Warning: Low quality match, the board may not be detected correctly")
    
    # Calculate board position
    minX, minY = minimum_location
    maxX = minX + small_image.shape[1]
    maxY = minY + small_image.shape[0]
    
    print(f"Board position: ({minX}, {minY}) to ({maxX}, {maxY})")
    position = Board_position(minX, minY, maxX, maxY)
    
    return position, we_are_white

def auto_find_chessboard():
    """Automatically find the chess board using edge and line detection"""
    print("Taking screenshot for auto detection...")
    img = take_screenshot()
    
    if img is None:
        print("Failed to take screenshot")
        return None, False
    
    print("Searching for chess board in screenshot...")
    is_found, current_chessboard_image, minX, minY, maxX, maxY, test_image = find_chessboard_from_image(img)
    
    if not is_found:
        print("Chess board could not be found automatically.")
        # Save the debug image if detection failed
        cv2.imwrite("chessboard_detection_failed.jpg", img)
        print("Saved debug image to chessboard_detection_failed.jpg")
        return None, False
    
    # Create board position object
    position = Board_position(minX, minY, maxX, maxY)
    
    # Determine board orientation
    is_white_bottom = is_white_on_bottom(current_chessboard_image)
    print(f"Detected board orientation: {'White' if is_white_bottom else 'Black'} pieces at bottom")
    
    # Save the detected board
    cv2.imwrite("chessboard_detection_result.jpg", test_image)
    print("Saved detected board to chessboard_detection_result.jpg")
    
    return position, is_white_bottom

def is_white_on_bottom(current_chessboard_image):
    """Determine if white pieces are on the bottom of the board"""
    try:
        # Get average brightness of bottom-left and top-right squares
        m1 = get_square_image(0, 0, current_chessboard_image).mean()
        m2 = get_square_image(7, 7, current_chessboard_image).mean()
        
        # In chess, bottom-left square (a1) is black, top-right (h8) is white
        # So if bottom-left is darker than top-right, white is playing from bottom
        return m1 < m2
    except Exception as e:
        print(f"Error determining board orientation: {e}")
        # Default to white on bottom
        return True

def get_square_image(row, column, board_img):
    """Extract a single square from the chess board image"""
    try:
        height, width = board_img.shape
        square_width = width // 8
        square_height = height // 8
        
        minX = column * square_width
        maxX = (column + 1) * square_width
        minY = row * square_height
        maxY = (row + 1) * square_height
        
        square = board_img[minY:maxY, minX:maxX]
        
        # Remove borders to focus on the square center
        if square.shape[0] > 6 and square.shape[1] > 6:  # Make sure square is big enough
            square_without_borders = square[3:-3, 3:-3]
            return square_without_borders
        else:
            return square
    except Exception as e:
        print(f"Error extracting square ({row},{column}): {e}")
        return np.zeros((1, 1), dtype=np.uint8)  # Return a tiny black square

def prepare(lines, kernel_close, kernel_open):
    """Prepare line image with morphological operations"""
    # Threshold to binary
    ret, lines = cv2.threshold(lines, 30, 255, cv2.THRESH_BINARY)
    
    # Close small gaps in lines
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel_close)
    
    # Open to remove noise
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, kernel_open)
    
    return lines

def prepare_vertical(lines):
    """Prepare vertical lines"""
    kernel_close = np.ones((3, 1), np.uint8)
    kernel_open = np.ones((50, 1), np.uint8)
    return prepare(lines, kernel_close, kernel_open)

def prepare_horizontal(lines):
    """Prepare horizontal lines"""
    kernel_close = np.ones((1, 3), np.uint8)
    kernel_open = np.ones((1, 50), np.uint8)
    return prepare(lines, kernel_close, kernel_open)

def find_chessboard_from_image(img):
    """Find chess board in image using line detection"""
    try:
        # Convert to grayscale for edge detection
        if len(img.shape) == 3:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            image = img.copy()
        
        # Create kernels for edge detection
        kernelH = np.array([[-1, 1]])
        kernelV = np.array([[-1], [1]])
        
        print("Detecting vertical lines...")
        # Find vertical lines
        vertical_lines = np.absolute(cv2.filter2D(image.astype('float'), -1, kernelH))
        image_vertical = prepare_vertical(vertical_lines)
        
        print("Detecting horizontal lines...")
        # Find horizontal lines
        horizontal_lines = np.absolute(cv2.filter2D(image.astype('float'), -1, kernelV))
        image_horizontal = prepare_horizontal(horizontal_lines)
        
        # Find line segments using Hough transform
        vertical_lines = cv2.HoughLinesP(image_vertical.astype(np.uint8), 1, np.pi / 180, 100, 
                                         minLineLength=100, maxLineGap=10)
        horizontal_lines = cv2.HoughLinesP(image_horizontal.astype(np.uint8), 1, np.pi / 180, 100, 
                                          minLineLength=100, maxLineGap=10)
        
        if vertical_lines is None or horizontal_lines is None:
            print("Failed to detect enough lines for a chess board")
            return False, image, 0, 0, 0, 0, image
        
        print(f"Found {len(vertical_lines)} vertical and {len(horizontal_lines)} horizontal lines")
        
        # Count intersections for each line
        v_count = [0 for _ in range(len(vertical_lines))]
        h_count = [0 for _ in range(len(horizontal_lines))]
        
        for i, line in enumerate(vertical_lines):
            x1, y1, x2, y2 = line[0]
            for j, other_line in enumerate(horizontal_lines):
                x3, y3, x4, y4 = other_line[0]
                # Check if lines intersect
                if ((x3 <= x1 <= x4) or (x4 <= x1 <= x3)) and ((y2 <= y3 <= y1) or (y1 <= y3 <= y2)):
                    v_count[i] += 1
                    h_count[j] += 1
        
        # Filter lines with enough intersections (likely part of the chess board)
        v_board = []
        h_board = []
        
        for i, line in enumerate(vertical_lines):
            if v_count[i] <= 6:  # Need at least 7 intersections for a board line
                continue
            v_board.append(line)

        for i, line in enumerate(horizontal_lines):
            if h_count[i] <= 6:
                continue
            h_board.append(line)
        
        print(f"After filtering: {len(v_board)} vertical and {len(h_board)} horizontal board lines")
        
        # If we have enough lines, calculate board boundaries
        if v_board and h_board:
            # Calculate board boundaries using median of line endpoints
            y_min = int(median(min(v[0][1], v[0][3]) for v in v_board))
            y_max = int(median(max(v[0][1], v[0][3]) for v in v_board))
            x_min = int(median(min(h[0][0], h[0][2]) for h in h_board))
            x_max = int(median(max(h[0][0], h[0][2]) for h in h_board))
            
            # Check if board is roughly square
            if abs((x_max - x_min) - (y_max - y_min)) > max((x_max - x_min), (y_max - y_min)) * 0.1:
                print("Board is not square enough.")
                print(f"Width: {x_max - x_min}, Height: {y_max - y_min}")
                return False, image, 0, 0, 0, 0, image
            
            # Extract the board
            board = image[y_min:y_max, x_min:x_max]
            
            # Resize to standard size
            dim = (800, 800)
            resized_board = cv2.resize(board, dim, interpolation=cv2.INTER_AREA)
            
            print(f"Chess board detected at ({x_min}, {y_min}) to ({x_max}, {y_max})")
            
            # Create a visualization of the detected board
            test_image = img.copy()
            if len(test_image.shape) == 2:  # If grayscale, convert to BGR
                test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
                
            # Draw board boundaries
            cv2.rectangle(test_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            return True, resized_board, int(x_min), int(y_min), int(x_max), int(y_max), test_image
        else:
            print("Not enough valid board lines found.")
            return False, image, 0, 0, 0, 0, image
            
    except Exception as e:
        print(f"Error in board detection: {e}")
        return False, image, 0, 0, 0, 0, image

# Test function to debug
def test_detection():
    """Test the chess board detection and show results"""
    try:
        # Try template matching
        print("\nTesting template matching...")
        position, we_are_white = find_chessboard()
        
        if position:
            print(f"Template matching successful: {'White' if we_are_white else 'Black'} side")
        else:
            print("Template matching failed")
        
        # Try auto detection
        print("\nTesting automatic detection...")
        auto_position, is_white_bottom = auto_find_chessboard()
        
        if auto_position:
            print(f"Auto detection successful: {'White' if is_white_bottom else 'Black'} on bottom")
            
            # Show the result
            result_img = cv2.imread("chessboard_detection_result.jpg")
            if result_img is not None:
                cv2.imshow("Detected Board", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Auto detection failed")
    
    except Exception as e:
        print(f"Error testing detection: {e}")

# Run test if script is executed directly
if __name__ == "__main__":
    test_detection()
