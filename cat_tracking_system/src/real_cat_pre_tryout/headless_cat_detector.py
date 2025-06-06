#!/usr/bin/env python3
"""
Headless Cat Detection for Raspberry Pi
Detects cats and returns ROI coordinates without GUI dependencies
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse
import logging

# Raspberry Pi camera imports
try:
    from picamera2 import Picamera2
    CAMERA_TYPE = "picamera2"
except ImportError:
    CAMERA_TYPE = "opencv"

class HeadlessCatDetector:
    def __init__(self, output_dir="detected_cats", confidence_threshold=0.5):
        """Initialize headless cat detector"""
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize camera
        self.camera = None
        self.init_camera()
        
        self.logger.info("Headless cat detector initialized")
    
    def init_camera(self):
        """Initialize camera for Raspberry Pi"""
        try:
            if CAMERA_TYPE == "picamera2":
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480)}
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Allow camera to warm up
                self.logger.info("PiCamera2 initialized successfully")
            else:
                # Fallback to OpenCV
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.logger.info("OpenCV camera initialized")
                
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
    
    def capture_frame(self):
        """Capture frame from camera"""
        try:
            if CAMERA_TYPE == "picamera2":
                frame = self.camera.capture_array()
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            else:
                ret, frame = self.camera.read()
                return frame if ret else None
                
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None
    
    def detect_cats_color_based(self, frame):
        """Detect cats using color and shape analysis"""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define color ranges for cat fur
        color_ranges = [
            # Orange/ginger cats
            ([8, 100, 100], [20, 255, 255]),
            # Gray cats
            ([0, 0, 40], [180, 50, 200]),
            # Brown/tan cats
            ([10, 50, 50], [25, 255, 255]),
            # Dark cats
            ([0, 0, 0], [180, 255, 100]),
            # Light cats
            ([0, 0, 180], [180, 40, 255]),
        ]
        
        # Create combined mask from all color ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (cats should be reasonably sized)
            if 1500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and dimensions
                aspect_ratio = w / float(h)
                if 0.4 < aspect_ratio < 2.5 and w > 50 and h > 50:
                    
                    # Additional shape analysis
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Cats tend to have moderate solidity (not too jagged)
                    if solidity > 0.6:
                        confidence = min(0.9, 0.4 + (solidity * 0.5))
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'Color+Shape',
                            'area': area,
                            'solidity': solidity
                        })
        
        # Sort by confidence and return top detections
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[:3]  # Return top 3 detections
    
    def detect_cats_edge_based(self, frame):
        """Alternative detection using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 40000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                if 0.5 < aspect_ratio < 2.0 and w > 60 and h > 60:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.5,
                        'method': 'Edge',
                        'area': area
                    })
        
        return detections
    
    def detect_cats(self, frame):
        """Main detection method combining multiple approaches"""
        # Try color-based detection first
        color_detections = self.detect_cats_color_based(frame)
        
        # If no good color detections, try edge-based
        if not color_detections or max([d['confidence'] for d in color_detections]) < 0.6:
            edge_detections = self.detect_cats_edge_based(frame)
            color_detections.extend(edge_detections)
        
        # Remove overlapping detections
        final_detections = self.remove_overlapping_detections(color_detections)
        
        # Filter by confidence threshold
        return [d for d in final_detections if d['confidence'] >= self.confidence_threshold]
    
    def remove_overlapping_detections(self, detections):
        """Remove overlapping bounding boxes"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        for detection in detections:
            x1, y1, w1, h1 = detection['bbox']
            
            # Check overlap with existing detections
            overlaps = False
            for existing in final_detections:
                x2, y2, w2, h2 = existing['bbox']
                
                # Calculate intersection over union (IoU)
                intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                                  max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                union_area = w1 * h1 + w2 * h2 - intersection_area
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > 0.3:  # 30% overlap threshold
                        overlaps = True
                        break
            
            if not overlaps:
                final_detections.append(detection)
        
        return final_detections
    
    def print_detections(self, detections):
        """Print detection results to console"""
        if detections:
            print(f"\n=== {len(detections)} Cat(s) Detected at {datetime.now().strftime('%H:%M:%S')} ===")
            for i, detection in enumerate(detections):
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                method = detection['method']
                
                print(f"Cat {i+1} ({method}):")
                print(f"  ROI Coordinates: x={x}, y={y}, width={w}, height={h}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Top-left: ({x}, {y}), Bottom-right: ({x+w}, {y+h})")
                
                if 'area' in detection:
                    print(f"  Area: {detection['area']} pixels")
                if 'solidity' in detection:
                    print(f"  Solidity: {detection['solidity']:.3f}")
                print()
        else:
            print(f"No cats detected at {datetime.now().strftime('%H:%M:%S')}")
    
    def save_roi_images(self, frame, detections):
        """Save ROI images of detected cats"""
        if not detections:
            return 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_count = 0
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Ensure ROI is within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w > 0 and h > 0:
                # Extract and save ROI
                roi = frame[y:y+h, x:x+w]
                filename = f"cat_roi_{timestamp}_{i}_conf{confidence:.2f}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                if cv2.imwrite(filepath, roi):
                    saved_count += 1
                    print(f"Saved: {filename} ({w}x{h})")
                else:
                    print(f"Failed to save: {filename}")
        
        # Also save full frame with annotations
        annotated_frame = self.draw_detections(frame, detections)
        full_filename = f"full_frame_{timestamp}.jpg"
        full_filepath = os.path.join(self.output_dir, full_filename)
        cv2.imwrite(full_filepath, annotated_frame)
        print(f"Saved annotated frame: {full_filename}")
        
        return saved_count
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        result = frame.copy()
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"Cat{i+1}: {confidence:.2f} ({method})"
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def run_continuous_detection(self, interval=2, save_images=False, max_runtime=None):
        """Run continuous detection"""
        print(f"Starting headless cat detection...")
        print(f"Detection interval: {interval} seconds")
        print(f"Save images: {'Yes' if save_images else 'No'}")
        if max_runtime:
            print(f"Max runtime: {max_runtime} seconds")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        detection_count = 0
        
        try:
            while True:
                # Check runtime limit
                if max_runtime and (time.time() - start_time) > max_runtime:
                    print("Max runtime reached. Stopping...")
                    break
                
                # Capture and process frame
                frame = self.capture_frame()
                if frame is None:
                    print("Failed to capture frame")
                    time.sleep(interval)
                    continue
                
                # Detect cats
                detections = self.detect_cats(frame)
                detection_count += 1
                
                # Print results
                self.print_detections(detections)
                
                # Save images if requested
                if save_images and detections:
                    saved = self.save_roi_images(frame, detections)
                    print(f"Saved {saved} ROI images")
                
                # Wait before next detection
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\nStopping detection... (Processed {detection_count} frames)")
        
        finally:
            self.cleanup()
    
    def single_detection(self, save_image=False):
        """Perform single detection"""
        print("Performing single cat detection...")
        
        frame = self.capture_frame()
        if frame is None:
            print("Failed to capture frame")
            return
        
        detections = self.detect_cats(frame)
        self.print_detections(detections)
        
        if save_image:
            if detections:
                saved = self.save_roi_images(frame, detections)
                print(f"Saved {saved} ROI images")
            else:
                # Save frame anyway for debugging
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"no_detection_{timestamp}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"Saved frame (no detections): {filename}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if CAMERA_TYPE == "picamera2" and self.camera:
                self.camera.stop()
                self.logger.info("PiCamera2 stopped")
            elif self.camera:
                self.camera.release()
                self.logger.info("Camera released")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Headless Cat ROI Detection for Raspberry Pi")
    parser.add_argument("--output-dir", default="detected_cats", 
                       help="Output directory for images")
    parser.add_argument("--confidence", type=float, default=0.4, 
                       help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--interval", type=int, default=2, 
                       help="Detection interval in seconds")
    parser.add_argument("--save", action="store_true", 
                       help="Save ROI images")
    parser.add_argument("--single", action="store_true", 
                       help="Single detection mode")
    parser.add_argument("--max-runtime", type=int, 
                       help="Maximum runtime in seconds")
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = HeadlessCatDetector(
            output_dir=args.output_dir,
            confidence_threshold=args.confidence
        )
        
        if args.single:
            # Single detection
            detector.single_detection(save_image=args.save)
        else:
            # Continuous detection
            detector.run_continuous_detection(
                interval=args.interval,
                save_images=args.save,
                max_runtime=args.max_runtime
            )
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())