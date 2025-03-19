import numpy as np
import cv2
from math import pi
import time

# https://github.com/youyexie/Chess-Piece-Recognition-using-Oriented-Chamfer-Matching-with-a-Comparison-to-CNN
class Classifier:
    def __init__(self, game_state):
        # Reduce resolution for better performance on Raspberry Pi
        self.dim = (320, 320)  # Smaller dimension for better performance
        
        # Resize the image with more efficient method
        self.img = cv2.resize(game_state.previous_chessboard_image, self.dim,
                              interpolation=cv2.INTER_NEAREST)
        
        # Calculate unit gradients
        self.img_x, self.img_y = self.unit_gradients(self.img)
        
        # Edge detection
        self.edges = cv2.Canny(self.img, 100, 200)
        self.inverted_edges = cv2.bitwise_not(self.edges)
        
        # Pre-compute distance transform (expensive operation)
        self.dist = cv2.distanceTransform(self.inverted_edges, cv2.DIST_L2, 3)
        
        # Precompute board representations for faster access
        print("Precomputing board representations...")
        start_time = time.time()
        
        # Use more efficient list comprehensions
        self.dist_board = [
            [self.get_square_image(row, column, self.dist) 
             for column in range(8)] 
            for row in range(8)
        ]
        
        self.edge_board = [
            [self.get_square_image(row, column, self.edges) 
             for column in range(8)] 
            for row in range(8)
        ]
        
        self.gradient_x = [
            [self.get_square_image(row, column, self.img_x) 
             for column in range(8)] 
            for row in range(8)
        ]
        
        self.gradient_y = [
            [self.get_square_image(row, column, self.img_y) 
             for column in range(8)] 
            for row in range(8)
        ]
        
        print(f"Board representations computed in {time.time() - start_time:.2f} seconds")

        # Helper function to measure edge intensity
        def intensity(x):
            return self.edge_board[x[0]][x[1]].mean()

        # Find templates with highest edge intensity
        pawn_templates = [
            max([(1, i) for i in range(8)], key=intensity),
            max([(6, i) for i in range(8)], key=intensity)
        ]

        # Create templates for all piece types
        self.templates = [pawn_templates] + [[(0, i), (7, i)] for i in range(5)]

        # Adjust knight templates if needed based on edge intensity
        if intensity((0, 6)) > intensity((0, 1)):
            self.templates[2][0] = (0, 6)

        if intensity((7, 6)) > intensity((7, 1)):
            self.templates[2][1] = (7, 6)

        # Piece symbols
        self.piece_symbol = [".", "p", "r", "n", "b", "q", "k"]
        
        # Swap queen and king if playing black
        if game_state.we_play_white == False:
            self.piece_symbol[-1], self.piece_symbol[-2] = self.piece_symbol[-2], self.piece_symbol[-1]
        
        # Create cache for classification results
        self.classification_cache = {}
        
    def classify(self, img):
        """Classify chess pieces in the given board image"""
        # Check cache using image hash (simple hash based on downsampled image)
        img_small = cv2.resize(img, (32, 32))
        img_hash = hash(img_small.tobytes())
        
        if img_hash in self.classification_cache:
            return self.classification_cache[img_hash]
        
        # Start timing
        start_time = time.time()
        
        # Resize with faster method for Pi
        img = cv2.resize(img, self.dim, interpolation=cv2.INTER_NEAREST)

        # Calculate gradients and edge maps
        img_x, img_y = self.unit_gradients(img)
        edges = cv2.Canny(img, 100, 200)
        inverted_edges = cv2.bitwise_not(edges)
        dist = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 3)
        
        # Process each square
        dist_board = [
            [self.get_square_image(row, column, dist) 
             for column in range(8)] 
            for row in range(8)
        ]
        
        gradient_x = [
            [self.get_square_image(row, column, img_x) 
             for column in range(8)] 
            for row in range(8)
        ]
        
        gradient_y = [
            [self.get_square_image(row, column, img_y) 
             for column in range(8)] 
            for row in range(8)
        ]

        # Initialize result grid
        result = []
        
        # Process each row
        for row in range(8):
            row_result = []
            for col in range(8):
                d = dist_board[row][col]
                template_scores = []
                
                # Compare against each piece template
                for piece in self.templates:
                    piece_scores = []
                    for tr, tc in piece:
                        # Get template
                        t = self.edge_board[tr][tc]
                        e = t / 255.0
                        e_c = e.sum()
                        
                        if e_c == 0:  # Avoid division by zero
                            piece_scores.append(float('inf'))
                            continue
                        
                        # Distance score
                        r_d = np.multiply(d, e).sum() / e_c

                        # Angle score - computationally expensive part
                        dp = np.multiply(self.gradient_x[tr][tc], gradient_x[row][col]) + \
                             np.multiply(self.gradient_y[tr][tc], gradient_y[row][col])
                        
                        # Clamp values for numerical stability
                        dp = np.clip(np.abs(dp), 0, 1.0)
                        
                        # Calculate angle difference
                        with np.errstate(invalid='ignore'):  # Suppress warnings
                            angle_difference = np.arccos(dp)
                            angle_difference = np.nan_to_num(angle_difference)  # Replace NaNs with 0
                        
                        r_o = np.multiply(angle_difference, e).sum() / (e_c * (pi / 2))
                        
                        # Combined score
                        piece_scores.append(r_d * 0.5 + r_o * 0.5)
                    
                    # Take minimum score for this piece type
                    if piece_scores:
                        template_scores.append(min(piece_scores))
                    else:
                        template_scores.append(float('inf'))
                
                # Find best matching piece (minimum score)
                min_score = float("inf")
                min_index = -1
                for i in range(len(template_scores)):
                    if min_score > template_scores[i]:
                        min_score = template_scores[i]
                        min_index = i
                
                # Assign piece or empty based on score threshold
                if min_score < 2.0:
                    row_result.append(self.piece_symbol[min_index + 1])
                else:
                    row_result.append(self.piece_symbol[0])

            result.append(row_result)
            
        # Cache the result
        self.classification_cache[img_hash] = result
        
        # Report processing time if it's slow
        elapsed = time.time() - start_time
        if elapsed > 0.5:
            print(f"Piece classification took {elapsed:.2f} seconds")
            
        return result

    def unit_gradients(self, gray):
        """Calculate unit gradient vectors for the image"""
        # Use smaller kernel for better performance
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and direction
        mag, direction = cv2.cartToPolar(sobelx, sobely)
        
        # Avoid division by zero
        mag[mag < 0.0001] = 0.0001
        
        # Calculate unit vectors
        unit_x = sobelx / mag
        unit_y = sobely / mag
        
        return unit_x, unit_y

    def get_square_image(self, row, column, board_img):
        """Extract a square from the chess board image"""
        height, width = board_img.shape[:2]
        
        # Calculate square boundaries
        minX = int(column * width / 8)
        maxX = int((column + 1) * width / 8)
        minY = int(row * height / 8)
        maxY = int((row + 1) * height / 8)
        
        # Extract square
        square = board_img[minY:maxY, minX:maxX]
        
        # Remove borders for better feature extraction
        border = 3
        
        # Make sure square is big enough to remove borders
        if square.shape[0] > 2*border and square.shape[1] > 2*border:
            square_without_borders = square[border:-border, border:-border]
            return square_without_borders
        else:
            return square  # Return full square if too small
