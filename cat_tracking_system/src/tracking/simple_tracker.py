"""
Simple tracker module for maintaining object identity across frames
"""
import logging
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional

# Import Detection class
from src.detection.yolo_detector import Detection


class TrackedObject:
    """Class to represent a tracked object with persistent ID"""
    
    def __init__(self, detection: Detection, track_id: int):
        """
        Initialize a tracked object
        
        Args:
            detection: Initial detection
            track_id: Unique tracking ID
        """
        # Basic properties
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.first_detected = time.time()
        self.last_detected = time.time()
        self.total_visible_time = 0
        self.is_active = True
        
        # Position and movement
        self.box = detection.box
        self.confidence = detection.confidence
        self.center = self._get_center(detection.box)
        self.velocity = [0, 0]  # [vx, vy] - pixels per frame
        
        # Detection history
        self.history = [detection.box]
        self.max_history = 30  # Keep track of last 30 positions
        
        # Appearance features (for identification)
        self.features = {}
    
    def update(self, detection: Detection):
        """
        Update the tracked object with a new detection
        
        Args:
            detection: New detection matching this tracked object
        """
        # Calculate time visible
        current_time = time.time()
        time_diff = current_time - self.last_detected
        self.total_visible_time += time_diff
        
        # Update timestamps
        self.last_detected = current_time
        
        # Get current center
        old_center = self.center
        
        # Update properties from detection
        self.box = detection.box
        self.confidence = detection.confidence
        self.center = self._get_center(detection.box)
        
        # Update velocity
        if len(self.history) > 0:
            dt = 1.0  # Assuming constant frame rate
            self.velocity = [
                (self.center[0] - old_center[0]) / dt,
                (self.center[1] - old_center[1]) / dt
            ]
        
        # Add to history
        self.history.append(detection.box)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def predict_next_position(self) -> Tuple[float, float]:
        """
        Predict the next position based on current velocity
        
        Returns:
            Predicted center (x, y)
        """
        return (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1]
        )
    
    def _get_center(self, box: List[float]) -> Tuple[float, float]:
        """
        Calculate center point of a bounding box
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Center coordinates (x, y)
        """
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class SimpleTracker:
    """
    Simple object tracker that maintains identity across frames 
    using IoU (Intersection over Union) matching.
    """
    
    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        """
        Initialize the tracker
        
        Args:
            max_disappeared: Maximum number of frames an object can disappear before being removed
            iou_threshold: IoU threshold for matching detections to tracked objects
        """
        self.logger = logging.getLogger(__name__)
        self.next_track_id = 0
        self.tracked_objects = {}  # Dictionary of tracked objects {track_id: TrackedObject}
        self.disappeared = {}  # Dictionary of how long each tracked object has disappeared
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
    
    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of new detections
            
        Returns:
            Dictionary of tracked objects
        """
        # If no tracked objects yet, create new ones for all detections
        if not self.tracked_objects:
            return self._init_tracks(detections)
        
        # If no detections, increment disappeared counters
        if not detections:
            return self._handle_disappeared()
        
        # Match detections to existing tracks
        return self._update_tracks(detections)
    
    def _init_tracks(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Initialize tracks for first-time detections"""
        for detection in detections:
            # Only track cats or specified classes
            if detection.class_name == "cat":
                track_id = self.next_track_id
                self.tracked_objects[track_id] = TrackedObject(detection, track_id)
                self.disappeared[track_id] = 0
                self.next_track_id += 1
        
        return self.tracked_objects
    
    def _handle_disappeared(self) -> Dict[int, TrackedObject]:
        """Handle case where no detections are made"""
        # Increment disappeared counter for all tracks
        ids_to_remove = []
        
        for track_id in self.tracked_objects:
            self.disappeared[track_id] += 1
            
            # Mark tracks for removal if disappeared for too long
            if self.disappeared[track_id] > self.max_disappeared:
                ids_to_remove.append(track_id)
                self.tracked_objects[track_id].is_active = False
        
        # Remove tracks that have disappeared for too long
        for track_id in ids_to_remove:
            self.logger.debug(f"Removing track {track_id} - disappeared for too long")
            del self.tracked_objects[track_id]
            del self.disappeared[track_id]
        
        return self.tracked_objects
    
    def _update_tracks(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Match detections to existing tracks and update"""
        if not self.tracked_objects:
            return self._init_tracks(detections)
        
        # Calculate IoU between all detections and tracked objects
        iou_matrix = self._calculate_iou_matrix(detections)
        
        # Match detections to tracks using IoU
        matches, unmatched_tracks, unmatched_detections = self._match_detections(iou_matrix)
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            track_id = list(self.tracked_objects.keys())[track_idx]
            self.tracked_objects[track_id].update(detections[detection_idx])
            self.disappeared[track_id] = 0
        
        # Handle disappeared tracks
        for track_idx in unmatched_tracks:
            track_id = list(self.tracked_objects.keys())[track_idx]
            self.disappeared[track_id] += 1
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            # Only track cats
            if detections[detection_idx].class_name == "cat":
                track_id = self.next_track_id
                self.tracked_objects[track_id] = TrackedObject(detections[detection_idx], track_id)
                self.disappeared[track_id] = 0
                self.next_track_id += 1
        
        # Remove tracks that have disappeared for too long
        ids_to_remove = []
        for track_id, disappeared_count in self.disappeared.items():
            if disappeared_count > self.max_disappeared:
                ids_to_remove.append(track_id)
                self.tracked_objects[track_id].is_active = False
        
        for track_id in ids_to_remove:
            self.logger.debug(f"Removing track {track_id} - disappeared for too long")
            del self.tracked_objects[track_id]
            del self.disappeared[track_id]
        
        return self.tracked_objects
    
    def _calculate_iou_matrix(self, detections: List[Detection]) -> np.ndarray:
        """
        Calculate IoU between all detections and tracked objects
        
        Args:
            detections: List of new detections
            
        Returns:
            Matrix of IoU values [tracks x detections]
        """
        # Initialize IoU matrix
        num_tracks = len(self.tracked_objects)
        num_detections = len(detections)
        iou_matrix = np.zeros((num_tracks, num_detections))
        
        # Calculate IoU for each track-detection pair
        for t_idx, track_id in enumerate(self.tracked_objects):
            track = self.tracked_objects[track_id]
            track_box = track.box
            
            for d_idx, detection in enumerate(detections):
                detection_box = detection.box
                iou_matrix[t_idx, d_idx] = self._calculate_iou(track_box, detection_box)
        
        return iou_matrix
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Handle case where boxes don't overlap
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        intersection_area = width * height
        
        # Calculate box areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _match_detections(self, iou_matrix: np.ndarray) -> Tuple[List, List, List]:
        """
        Match detections to tracks using IoU matrix
        
        Args:
            iou_matrix: Matrix of IoU values [tracks x detections]
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        # Initialize lists
        matches = []
        unmatched_tracks = []
        unmatched_detections = []
        
        # Get matrix dimensions
        num_tracks, num_detections = iou_matrix.shape
        
        # Handle edge cases
        if num_tracks == 0:
            unmatched_detections = list(range(num_detections))
            return matches, unmatched_tracks, unmatched_detections
        
        if num_detections == 0:
            unmatched_tracks = list(range(num_tracks))
            return matches, unmatched_tracks, unmatched_detections
        
        # Find matches
        # Greedy matching algorithm - match highest IoU pairs first
        while iou_matrix.size > 0 and iou_matrix.max() >= self.iou_threshold:
            # Find maximum IoU
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, detection_idx = max_idx
            
            # Add to matches
            matches.append((track_idx, detection_idx))
            
            # Remove matched row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, detection_idx] = 0
        
        # Find unmatched tracks and detections
        matched_track_indices = [match[0] for match in matches]
        matched_detection_indices = [match[1] for match in matches]
        
        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(num_detections) if i not in matched_detection_indices]
        
        return matches, unmatched_tracks, unmatched_detections