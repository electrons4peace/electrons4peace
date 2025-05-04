"""
Feature extractor for cat identification
"""
import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Optional


class FeatureExtractor:
    """
    Extracts visual features from cat images for identification.
    Uses a combination of color histograms and basic pattern features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Feature extraction settings
        self.hist_bins = config.get("hist_bins", 32)
        self.pattern_cells = config.get("pattern_cells", 4)  # 4x4 grid
        self.feature_size = self.hist_bins * 3 + self.pattern_cells * self.pattern_cells * 2
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a cat image
        
        Args:
            image: Image of a cat (cropped to bounding box)
            
        Returns:
            Feature vector as numpy array
        """
        if image is None or image.size == 0:
            self.logger.warning("Empty image provided to feature extractor")
            return np.zeros(self.feature_size)
        
        try:
            # Resize image for consistent features
            img_resized = cv2.resize(image, (128, 128))
            
            # Extract color histograms
            color_features = self._extract_color_features(img_resized)
            
            # Extract pattern features
            pattern_features = self._extract_pattern_features(img_resized)
            
            # Combine features
            features = np.concatenate([color_features, pattern_features])
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(self.feature_size)
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features
        
        Args:
            image: Resized cat image
            
        Returns:
            Color feature vector
        """
        # Convert to HSV color space (better for color analysis)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        hist_h = cv2.calcHist([hsv_image], [0], None, [self.hist_bins], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [self.hist_bins], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [self.hist_bins], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
        
        # Flatten and combine
        color_features = np.concatenate([
            hist_h.flatten(), 
            hist_s.flatten(), 
            hist_v.flatten()
        ])
        
        return color_features
    
    def _extract_pattern_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract simple pattern features using gradients
        
        Args:
            image: Resized cat image
            
        Returns:
            Pattern feature vector
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients using Sobel filters
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and orientation
        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # Calculate cell size
        h, w = gray.shape
        cell_h = h // self.pattern_cells
        cell_w = w // self.pattern_cells
        
        # Initialize feature vector
        pattern_features = np.zeros(self.pattern_cells * self.pattern_cells * 2)
        
        # Calculate features for each cell
        index = 0
        for i in range(self.pattern_cells):
            for j in range(self.pattern_cells):
                # Get cell coordinates
                y_start = i * cell_h
                y_end = (i + 1) * cell_h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w
                
                # Extract cell from magnitude and orientation
                cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
                cell_orientation = orientation[y_start:y_end, x_start:x_end]
                
                # Calculate average magnitude and orientation
                avg_magnitude = np.mean(cell_magnitude)
                avg_orientation = np.mean(cell_orientation)
                
                # Add to feature vector
                pattern_features[index] = avg_magnitude
                pattern_features[index + 1] = avg_orientation
                index += 2
        
        # Normalize features
        pattern_features = pattern_features / (np.max(pattern_features) + 1e-8)
        
        return pattern_features
    
    def compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compare two feature vectors and return similarity
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure features are numpy arrays
        if not isinstance(features1, np.ndarray):
            features1 = np.array(features1)
        if not isinstance(features2, np.ndarray):
            features2 = np.array(features2)
        
        # Check for empty features
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(features1, features2)
        
        # Ensure value is in range [0,1]
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)