#!/usr/bin/env python3
"""
Real-time Cat Detection with ROI (Region of Interest)
Detects cats and returns bounding box coordinates (x, y, width, height)
Supports both YOLO and OpenCV Haar Cascade methods
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse
import logging

# Try different camera libraries based on what's available
try:
    from picamera2 import Picamera2
    CAMERA_TYPE = "picamera2"
except ImportError:
    try:
        import picamera
        CAMERA_TYPE = "picamera"
    except ImportError:
        CAMERA_TYPE = "opencv"

class CatROIDetector:
    def __init__(self, model_path="yolo_models", output_dir="detected_cats", 
                 confidence_threshold=0.5, detection_method="yolo"):
        """
        Initialize the cat detector
        
        Args:
            model_path: Path to YOLO model files
            output_dir: Directory to save detected regions
            confidence_threshold: Minimum confidence for detections
            detection_method: "yolo" or "haar" for detection method
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.detection_method = detection_method
        self.nms_threshold = 0.4
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize camera
        self.camera = None
        self.init_camera()
        
        # Initialize detection method
        if detection_method == "yolo":
            self.init_yolo()
        else:
            self.init_haar_cascade()
    
    def init_camera(self):
        """Initialize the appropriate camera based on available libraries"""
        try:
            if CAMERA_TYPE == "picamera2":
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480)}
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)
                self.logger.info("Initialized PiCamera2")
                
            elif CAMERA_TYPE == "picamera":
                import picamera
                self.camera = picamera.PiCamera()
                self.camera.resolution = (640, 480)
                self.camera.start_preview()
                time.sleep(2)
                self.logger.info("Initialized PiCamera")
                
            else:
                # OpenCV camera
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.logger.info("Initialized OpenCV camera")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def init_yolo(self):
        """Initialize YOLO model for cat detection"""
        try:
            # YOLO file paths
            weights_path = os.path.join(self.model_path, "yolov4.weights")
            config_path = os.path.join(self.model_path, "yolov4.cfg")
            classes_path = os.path.join(self.model_path, "coco.names")
            
            # Check if files exist
            if not all(os.path.exists(p) for p in [weights_path, config_path, classes_path]):
                self.logger.warning("YOLO model files not found. Falling back to Haar Cascade.")
                self.detection_method = "haar"
                self.init_haar_cascade()
                return
            
            # Load YOLO
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(classes_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.logger.info("YOLO model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.detection_method = "haar"
            self.init_haar_cascade()
    
    def init_haar_cascade(self):
        """Initialize Haar Cascade for cat detection"""
        try:
            # Try to load cat face cascade (you might need to download this)
            cat_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
            
            if os.path.exists(cat_cascade_path):
                self.cat_cascade = cv2.CascadeClassifier(cat_cascade_path)
                self.logger.info("Loaded cat face Haar cascade")
            else:
                # Fallback to general object detection or download instructions
                self.logger.warning("Cat cascade not found. You may need to download it.")
                self.logger.info("Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
                # Use a general approach as fallback
                self.cat_cascade = None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Haar cascade: {e}")
            self.cat_cascade = None
    
    def capture_frame(self):
        """Capture a single frame from camera"""
        try:
            if CAMERA_TYPE == "picamera2":
                frame = self.camera.capture_array()
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
            elif CAMERA_TYPE == "picamera":
                import io
                stream = io.BytesIO()
                self.camera.capture(stream, format='jpeg')
                data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                
            else:
                ret, frame = self.camera.read()
                if not ret:
                    return None
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None
    
    def detect_cats_yolo(self, frame):
        """Detect cats using YOLO"""
        if not hasattr(self, 'net') or self.net is None:
            return []
        
        height, width, channels = frame.shape
        
        # Prepare frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Check if it's a cat (class_id 15 in COCO dataset)
                if class_id == 15 and confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'method': 'YOLO'
                })
        
        return detections
    
    def detect_cats_haar(self, frame):
        """Detect cats using Haar Cascade"""
        if self.cat_cascade is None:
            return []
        
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect cat faces
        cats = self.cat_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in cats:
            # Expand bounding box to include more of the cat's body
            # This is an estimation since Haar cascade typically detects faces
            expanded_h = int(h * 2.5)  # Assume body is ~2.5x the face height
            expanded_w = int(w * 1.5)   # Expand width slightly
            expanded_x = max(0, x - int((expanded_w - w) / 2))
            expanded_y = max(0, y - int((expanded_h - h) / 4))  # Keep face in upper portion
            
            # Ensure we don't go outside frame boundaries
            expanded_w = min(expanded_w, frame.shape[1] - expanded_x)
            expanded_h = min(expanded_h, frame.shape[0] - expanded_y)
            
            detections.append({
                'bbox': (expanded_x, expanded_y, expanded_w, expanded_h),
                'confidence': 0.8,  # Fixed confidence for Haar cascade
                'method': 'Haar'
            })
        
        return detections
    
    def detect_cats(self, frame):
        """Main detection method"""
        if self.detection_method == "yolo":
            return self.detect_cats_yolo(frame)
        else:
            return self.detect_cats_haar(frame)
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and ROI information on frame"""
        result_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label with ROI coordinates
            label = f"Cat {i+1} ({method}): {confidence:.2f}"
            roi_text = f"ROI: ({x}, {y}, {w}, {h})"
            
            cv2.putText(result_frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_frame, roi_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_frame
    
    def print_roi_coordinates(self, detections):
        """Print ROI coordinates to console"""
        if detections:
            print(f"\n--- Detected {len(detections)} cat(s) at {datetime.now().strftime('%H:%M:%S')} ---")
            for i, detection in enumerate(detections):
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                method = detection['method']
                print(f"Cat {i+1} ({method}):")
                print(f"  ROI Coordinates: x={x}, y={y}, width={w}, height={h}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Bounding Box: ({x}, {y}) to ({x+w}, {y+h})")
        else:
            print("No cats detected")
    
    def save_roi_image(self, frame, detections, timestamp):
        """Save cropped ROI images of detected cats"""
        saved_count = 0
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Ensure ROI is within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w > 0 and h > 0:
                # Extract ROI
                roi = frame[y:y+h, x:x+w]
                
                # Save ROI
                filename = f"cat_roi_{timestamp}_{i}_{confidence:.2f}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, roi)
                saved_count += 1
                
                print(f"Saved ROI: {filename}")
        
        return saved_count
    
    def run_realtime_detection(self, show_video=True, save_detections=False, print_coords=True):
        """Run real-time cat detection"""
        print(f"\nStarting real-time cat detection using {self.detection_method.upper()} method...")
        print("Press 'q' to quit, 's' to save current detections")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Detect cats
                detections = self.detect_cats(frame)
                
                # Print coordinates if requested
                if print_coords and detections:
                    self.print_roi_coordinates(detections)
                
                # Draw detections on frame
                if show_video:
                    display_frame = self.draw_detections(frame, detections)
                    
                    # Calculate and display FPS
                    frame_count += 1
                    if frame_count % 30 == 0:  # Update FPS every 30 frames
                        fps = 30 / (time.time() - fps_start_time)
                        fps_start_time = time.time()
                    else:
                        fps = 0
                    
                    if fps > 0:
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show frame
                    cv2.imshow("Cat ROI Detection", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_detections and detections:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved = self.save_roi_image(frame, detections, timestamp)
                    print(f"Saved {saved} ROI image(s)")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if CAMERA_TYPE == "picamera2" and self.camera:
                self.camera.stop()
            elif CAMERA_TYPE == "picamera" and self.camera:
                self.camera.stop_preview()
                self.camera.close()
            elif CAMERA_TYPE == "opencv" and self.camera:
                self.camera.release()
            
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Real-time Cat ROI Detection")
    parser.add_argument("--model-path", default="yolo_models", help="Path to YOLO model files")
    parser.add_argument("--output-dir", default="detected_cats", help="Output directory for ROI images")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--method", choices=["yolo", "haar"], default="yolo", 
                       help="Detection method: yolo or haar")
    parser.add_argument("--no-video", action="store_true", help="Don't show video window")
    parser.add_argument("--save", action="store_true", help="Enable saving ROI images")
    parser.add_argument("--no-coords", action="store_true", help="Don't print coordinates")
    
    args = parser.parse_args()
    
    # Create detector
    detector = CatROIDetector(
        model_path=args.model_path,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        detection_method=args.method
    )
    
    # Run detection
    detector.run_realtime_detection(
        show_video=not args.no_video,
        save_detections=args.save,
        print_coords=not args.no_coords
    )

if __name__ == "__main__":
    main()