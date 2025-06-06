#!/usr/bin/env python3
"""
MOG2-based Surveillance System with ROI Detection
Uses background images to initialize MOG2 for better detection
Handles dual cameras (NoIR for low light, Standard for bright conditions)
Detects and tracks stuffed animals added to the scene
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict
from picamera2 import Picamera2

@dataclass
class TrackedObject:
    """Class to track detected objects across frames"""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    last_seen: int
    confidence: float
    history: deque  # History of positions

class DualCameraSurveillance:
    def __init__(self, background_dir="background_images", output_dir="surveillance_detections", 
                 min_area=500, max_area=50000, history_length=30):
        """
        Initialize surveillance system
        
        Args:
            background_dir: Directory containing background images
            output_dir: Directory to save detections
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            history_length: Number of frames to track object history
        """
        self.background_dir = background_dir
        self.output_dir = output_dir
        self.min_area = min_area
        self.max_area = max_area
        self.history_length = history_length
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.noir_output = os.path.join(output_dir, "noir_detections")
        self.standard_output = os.path.join(output_dir, "standard_detections")
        os.makedirs(self.noir_output, exist_ok=True)
        os.makedirs(self.standard_output, exist_ok=True)
        
        # Initialize cameras
        self.camera_noir = None
        self.camera_standard = None
        
        # MOG2 background subtractors
        self.mog2_noir = None
        self.mog2_standard = None
        
        # Object tracking
        self.next_object_id = 0
        self.tracked_objects_noir = {}
        self.tracked_objects_standard = {}
        
        # Frame counters
        self.frame_count_noir = 0
        self.frame_count_standard = 0
        
        # Detection statistics
        self.stats = {
            'noir': {'detections': 0, 'saved': 0},
            'standard': {'detections': 0, 'saved': 0}
        }
    
    def init_cameras(self):
        """Initialize both Pi Camera modules"""
        try:
            # Initialize NoIR camera
            self.logger.info("Initializing NoIR camera...")
            self.camera_noir = Picamera2(0)
            config_noir = self.camera_noir.create_preview_configuration(
                main={"size": (1280, 720)},  # Lower res for real-time processing
                buffer_count=1
            )
            self.camera_noir.configure(config_noir)
            
            # Set NoIR camera for low light
            self.camera_noir.set_controls({
                "AnalogueGain": 8.0,  # Higher gain for low light
                "ExposureTime": 40000  # Longer exposure for low light
            })
            
            self.camera_noir.start()
            time.sleep(2)
            self.logger.info("NoIR camera initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NoIR camera: {e}")
            self.camera_noir = None
        
        try:
            # Initialize standard camera
            self.logger.info("Initializing standard camera...")
            self.camera_standard = Picamera2(1)
            config_standard = self.camera_standard.create_preview_configuration(
                main={"size": (1280, 720)},
                buffer_count=1
            )
            self.camera_standard.configure(config_standard)
            
            # Set standard camera for bright conditions
            self.camera_standard.set_controls({
                "AnalogueGain": 1.0,  # Lower gain for bright light
                "ExposureTime": 10000  # Shorter exposure for bright light
            })
            
            self.camera_standard.start()
            time.sleep(2)
            self.logger.info("Standard camera initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize standard camera: {e}")
            self.camera_standard = None
        
        if self.camera_noir is None and self.camera_standard is None:
            raise Exception("No cameras could be initialized!")
    
    def init_mog2_with_background(self, camera_type="noir"):
        """Initialize MOG2 with background images"""
        # Create MOG2 with optimized parameters for stationary camera
        mog2 = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,  # Sensitivity threshold
            history=500  # Number of frames for background model
        )
        
        # Set additional parameters
        mog2.setShadowValue(127)  # Shadow pixel value
        mog2.setShadowThreshold(0.5)  # Shadow threshold
        
        # Load background images
        if camera_type == "noir":
            bg_path = os.path.join(self.background_dir, "camera1_noir")
        else:
            bg_path = os.path.join(self.background_dir, "camera2_standard")
        
        if os.path.exists(bg_path):
            bg_files = sorted([f for f in os.listdir(bg_path) if f.endswith('.jpg')])[:20]
            
            self.logger.info(f"Training {camera_type} MOG2 with {len(bg_files)} background images...")
            
            for i, bg_file in enumerate(bg_files):
                img_path = os.path.join(bg_path, bg_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize to match camera resolution
                    img_resized = cv2.resize(img, (1280, 720))
                    
                    # Apply MOG2 learning
                    mog2.apply(img_resized, learningRate=0.1)
                    
                    if i % 5 == 0:
                        self.logger.info(f"Processed {i+1}/{len(bg_files)} background images")
            
            self.logger.info(f"MOG2 training complete for {camera_type}")
        else:
            self.logger.warning(f"No background images found for {camera_type}, starting fresh")
        
        return mog2
    
    def preprocess_frame(self, frame, camera_type="noir"):
        """Preprocess frame for better detection"""
        # Apply different preprocessing based on camera type
        if camera_type == "noir":
            # For low light NoIR camera
            # Enhance contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Reduce noise
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
            
        else:
            # For bright standard camera
            # Simple gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return blurred
    
    def detect_objects(self, frame, mog2, camera_type="noir"):
        """Detect objects using MOG2 background subtraction"""
        # Preprocess frame
        processed = self.preprocess_frame(frame, camera_type)
        
        # Apply MOG2
        fg_mask = mog2.apply(processed, learningRate=0.001)
        
        # Remove shadows (shadows have value 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area < area < self.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (stuffed animals are usually not too elongated)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 3.0:
                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate confidence based on area and solidity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    confidence = min(solidity * (area / self.max_area), 1.0)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (center_x, center_y),
                        'area': area,
                        'confidence': confidence,
                        'contour': contour
                    })
        
        return detections, fg_mask
    
    def track_objects(self, detections, tracked_objects, camera_type):
        """Track objects across frames"""
        # Distance threshold for matching (in pixels)
        distance_threshold = 50
        
        # Mark all existing objects as not seen
        for obj_id in tracked_objects:
            tracked_objects[obj_id].last_seen += 1
        
        # Match detections to existing objects
        matched = set()
        
        for detection in detections:
            center = detection['center']
            best_match = None
            best_distance = float('inf')
            
            # Find closest existing object
            for obj_id, obj in tracked_objects.items():
                if obj_id not in matched:
                    # Calculate distance between centers
                    dist = np.sqrt((center[0] - obj.center[0])**2 + 
                                 (center[1] - obj.center[1])**2)
                    
                    if dist < distance_threshold and dist < best_distance:
                        best_match = obj_id
                        best_distance = dist
            
            if best_match is not None:
                # Update existing object
                obj = tracked_objects[best_match]
                obj.bbox = detection['bbox']
                obj.center = center
                obj.last_seen = 0
                obj.confidence = detection['confidence']
                obj.history.append(center)
                matched.add(best_match)
            else:
                # Create new object
                new_obj = TrackedObject(
                    id=self.next_object_id,
                    bbox=detection['bbox'],
                    center=center,
                    last_seen=0,
                    confidence=detection['confidence'],
                    history=deque([center], maxlen=self.history_length)
                )
                tracked_objects[self.next_object_id] = new_obj
                self.next_object_id += 1
                
                # Log new detection
                self.logger.info(f"New object detected on {camera_type} camera: ID {new_obj.id}")
                self.stats[camera_type]['detections'] += 1
        
        # Remove objects not seen for too long
        to_remove = []
        for obj_id, obj in tracked_objects.items():
            if obj.last_seen > 30:  # 30 frames = ~1 second at 30fps
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del tracked_objects[obj_id]
            self.logger.info(f"Lost object on {camera_type} camera: ID {obj_id}")
        
        return tracked_objects
    
    def draw_detections(self, frame, tracked_objects, fg_mask, camera_type):
        """Draw bounding boxes and information on frame"""
        result_frame = frame.copy()
        
        # Draw each tracked object
        for obj_id, obj in tracked_objects.items():
            x, y, w, h = obj.bbox
            
            # Color based on confidence
            color = (0, int(255 * obj.confidence), int(255 * (1 - obj.confidence)))
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw object info
            label = f"ID: {obj.id} ({obj.confidence:.2f})"
            cv2.putText(result_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw movement trail
            if len(obj.history) > 1:
                points = np.array(list(obj.history), np.int32)
                cv2.polylines(result_frame, [points], False, color, 1)
        
        # Add camera type and stats
        info_text = f"{camera_type.upper()} - Objects: {len(tracked_objects)} - Total Detected: {self.stats[camera_type]['detections']}"
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create combined view with mask
        mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([result_frame, mask_colored])
        
        return combined
    
    def save_detection(self, frame, obj, camera_type):
        """Save detected object ROI"""
        x, y, w, h = obj.bbox
        
        # Ensure ROI is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w > 0 and h > 0:
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_dir = self.noir_output if camera_type == "noir" else self.standard_output
            filename = f"{camera_type}_obj{obj.id}_{timestamp}_conf{obj.confidence:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save ROI and metadata
            cv2.imwrite(filepath, roi)
            
            # Save metadata
            metadata_file = filepath.replace('.jpg', '_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f"Object ID: {obj.id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Camera: {camera_type}\n")
                f.write(f"Bounding Box: {x}, {y}, {w}, {h}\n")
                f.write(f"Confidence: {obj.confidence:.3f}\n")
                f.write(f"Center: {obj.center}\n")
            
            self.stats[camera_type]['saved'] += 1
            self.logger.info(f"Saved detection: {filename}")
    
    def process_camera(self, camera, mog2, tracked_objects, camera_type):
        """Process single camera frame"""
        if camera is None:
            return None
        
        try:
            # Capture frame
            frame = camera.capture_array()
            
            # Convert RGB to BGR
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect objects
            detections, fg_mask = self.detect_objects(frame, mog2, camera_type)
            
            # Track objects
            self.track_objects(detections, tracked_objects, camera_type)
            
            # Save new detections (only save when first detected and confidence is high)
            for obj_id, obj in tracked_objects.items():
                if obj.last_seen == 0 and len(obj.history) == 1 and obj.confidence > 0.7:
                    self.save_detection(frame, obj, camera_type)
            
            # Draw visualization
            result = self.draw_detections(frame, tracked_objects, fg_mask, camera_type)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {camera_type} camera: {e}")
            return None
    
    def run_surveillance(self, show_video=True, save_interval=300):
        """Run surveillance on both cameras"""
        self.logger.info("Starting dual camera surveillance system...")
        
        # Initialize cameras
        self.init_cameras()
        
        # Initialize MOG2 with background images
        if self.camera_noir:
            self.mog2_noir = self.init_mog2_with_background("noir")
        if self.camera_standard:
            self.mog2_standard = self.init_mog2_with_background("standard")
        
        self.logger.info("Surveillance system ready. Press 'q' to quit, 's' to force save current detections")
        
        fps_time = time.time()
        fps_counter = 0
        
        try:
            while True:
                # Process NoIR camera
                if self.camera_noir and self.mog2_noir:
                    result_noir = self.process_camera(
                        self.camera_noir, self.mog2_noir, 
                        self.tracked_objects_noir, "noir"
                    )
                    if show_video and result_noir is not None:
                        cv2.imshow("NoIR Camera Surveillance", cv2.resize(result_noir, (1280, 360)))
                
                # Process standard camera
                if self.camera_standard and self.mog2_standard:
                    result_standard = self.process_camera(
                        self.camera_standard, self.mog2_standard,
                        self.tracked_objects_standard, "standard"
                    )
                    if show_video and result_standard is not None:
                        cv2.imshow("Standard Camera Surveillance", cv2.resize(result_standard, (1280, 360)))
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_time)
                    fps_time = time.time()
                    self.logger.info(f"Processing at {fps:.1f} FPS")
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Force save all current detections
                    for obj_id, obj in self.tracked_objects_noir.items():
                        if self.camera_noir:
                            frame = self.camera_noir.capture_array()
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            self.save_detection(frame, obj, "noir")
                    
                    for obj_id, obj in self.tracked_objects_standard.items():
                        if self.camera_standard:
                            frame = self.camera_standard.capture_array()
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            self.save_detection(frame, obj, "standard")
                    
                    self.logger.info("Forced save of all current detections")
                
        except KeyboardInterrupt:
            self.logger.info("Surveillance stopped by user")
        
        finally:
            self.cleanup()
            self.print_statistics()
    
    def print_statistics(self):
        """Print surveillance statistics"""
        print("\n" + "="*50)
        print("SURVEILLANCE STATISTICS")
        print("="*50)
        
        for camera_type in ['noir', 'standard']:
            stats = self.stats[camera_type]
            print(f"\n{camera_type.upper()} Camera:")
            print(f"  Total objects detected: {stats['detections']}")
            print(f"  ROIs saved: {stats['saved']}")
        
        print(f"\nOutput directories:")
        print(f"  NoIR detections: {self.noir_output}")
        print(f"  Standard detections: {self.standard_output}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera_noir:
                self.camera_noir.stop()
                self.logger.info("NoIR camera stopped")
            
            if self.camera_standard:
                self.camera_standard.stop()
                self.logger.info("Standard camera stopped")
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="MOG2 Surveillance with dual cameras")
    parser.add_argument("--background-dir", default="background_images", 
                       help="Directory containing background images")
    parser.add_argument("--output-dir", default="surveillance_detections", 
                       help="Output directory for detections")
    parser.add_argument("--min-area", type=int, default=500, 
                       help="Minimum object area in pixels")
    parser.add_argument("--max-area", type=int, default=50000, 
                       help="Maximum object area in pixels")
    parser.add_argument("--no-video", action="store_true", 
                       help="Run without video display")
    
    args = parser.parse_args()
    
    # Create surveillance system
    surveillance = DualCameraSurveillance(
        background_dir=args.background_dir,
        output_dir=args.output_dir,
        min_area=args.min_area,
        max_area=args.max_area
    )
    
    print("MOG2 Surveillance System with Dual Cameras")
    print(f"Background images: {args.background_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Object size range: {args.min_area} - {args.max_area} pixels")
    print("-" * 50)
    
    surveillance.run_surveillance(show_video=not args.no_video)

if __name__ == "__main__":
    main()