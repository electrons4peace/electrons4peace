#!/usr/bin/env python3
"""
Stuffed Animal Detection using YOLO on Raspberry Pi
Captures images, detects stuffed animals, and saves regions of interest
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

class StuffedAnimalDetector:
    def __init__(self, model_path="yolo_models", output_dir="detected_animals", confidence_threshold=0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to YOLO model files
            output_dir: Directory to save detected regions
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize camera
        self.camera = None
        self.init_camera()
        
        # Load YOLO model
        self.net = None
        self.output_layers = None
        self.classes = None
        self.load_yolo_model()
    
    def init_camera(self):
        """Initialize the appropriate camera based on available libraries"""
        try:
            if CAMERA_TYPE == "picamera2":
                self.camera = Picamera2()
                # Configure camera for still capture
                config = self.camera.create_still_configuration(
                    main={"size": (1024, 768)},
                    lores={"size": (640, 480)},
                    display="lores"
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Allow camera to warm up
                self.logger.info("Initialized PiCamera2")
                
            elif CAMERA_TYPE == "picamera":
                import picamera
                self.camera = picamera.PiCamera()
                self.camera.resolution = (1024, 768)
                self.camera.start_preview()
                time.sleep(2)
                self.logger.info("Initialized PiCamera")
                
            else:
                # Fallback to OpenCV camera
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
                self.logger.info("Initialized OpenCV camera")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def load_yolo_model(self):
        """Load YOLO model files"""
        try:
            # YOLO file paths
            weights_path = os.path.join(self.model_path, "yolov4.weights")
            config_path = os.path.join(self.model_path, "yolov4.cfg")
            classes_path = os.path.join(self.model_path, "coco.names")
            
            # Check if files exist, if not, provide download instructions
            if not all(os.path.exists(p) for p in [weights_path, config_path, classes_path]):
                self.logger.warning("YOLO model files not found. Using basic detection.")
                self.setup_basic_detection()
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
            self.setup_basic_detection()
    
    def setup_basic_detection(self):
        """Setup basic color-based detection as fallback"""
        self.logger.info("Setting up basic color-based detection")
        self.classes = ["stuffed_animal"]  # Generic class for basic detection
    
    def capture_image(self):
        """Capture image from camera"""
        try:
            if CAMERA_TYPE == "picamera2":
                # Capture with PiCamera2
                image_array = self.camera.capture_array()
                # Convert from RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image = image_array
                    
            elif CAMERA_TYPE == "picamera":
                # Capture with PiCamera
                import io
                stream = io.BytesIO()
                self.camera.capture(stream, format='jpeg')
                data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                
            else:
                # Capture with OpenCV
                ret, image = self.camera.read()
                if not ret:
                    raise Exception("Failed to capture image")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            return None
    
    def detect_with_yolo(self, image):
        """Detect objects using YOLO"""
        if self.net is None:
            return self.detect_basic(image)
        
        height, width, channels = image.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                
                # Filter for toy-like objects and high confidence
                toy_classes = ["teddy bear", "toy", "doll", "bear"]
                class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
                
                # More strict filtering - only accept high confidence toy detections
                if (confidence > self.confidence_threshold and 
                    any(toy in class_name.lower() for toy in toy_classes) and
                    # Additional size filtering - stuffed animals should be reasonably sized
                    detection[2] * width > 50 and detection[3] * height > 50):
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
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "stuffed_animal"
                confidence = confidences[i]
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'class': class_name
                })
        
        # Post-process detections to remove likely false positives
        filtered_detections = self.filter_false_positives(image, detections)
        return filtered_detections
    
    def filter_false_positives(self, image, detections):
        """Filter out likely false positive detections"""
        filtered = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Extract the region
            roi = image[y:y+h, x:x+w]
            
            # Calculate color variance (stuffed animals usually have more color variation than floors)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            color_variance = np.var(roi_hsv)
            
            # Calculate edge density (stuffed animals have more edges than flat surfaces)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            # Filter criteria
            if (color_variance > 100 and  # Has color variation
                edge_density > 0.05 and   # Has sufficient edges
                w > 80 and h > 80):       # Reasonable size
                filtered.append(detection)
                self.logger.info(f"Kept detection: variance={color_variance:.1f}, edges={edge_density:.3f}")
            else:
                self.logger.info(f"Filtered out: variance={color_variance:.1f}, edges={edge_density:.3f}")
        
        return filtered
    
    def detect_basic(self, image):
        """Enhanced color-based detection for your specific stuffed animals"""
        self.logger.info("Using enhanced color-based detection")
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for your specific stuffed animals (based on image)
        color_ranges = [
            # Yellow (like your yellow toy)
            ([15, 100, 100], [35, 255, 255]),
            # Blue/cyan (like your blue toy)
            ([85, 100, 100], [125, 255, 255]),
            # Pink/red (like your red toy)
            ([0, 100, 100], [15, 255, 255]),
            ([165, 100, 100], [180, 255, 255]),
            # White areas of toys
            ([0, 0, 200], [180, 30, 255]),
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Larger minimum area
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional shape filtering
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.8,
                        'class': 'stuffed_animal'
                    })
        
        return detections
    
    def save_detections(self, image, detections, timestamp):
        """Save detected regions as separate images"""
        saved_count = 0
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Ensure bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                # Extract region of interest
                roi = image[y:y+h, x:x+w]
                
                # Create filename
                filename = f"{timestamp}_{i}_{class_name}_{confidence:.2f}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save ROI
                cv2.imwrite(filepath, roi)
                saved_count += 1
                
                self.logger.info(f"Saved: {filename} ({w}x{h})")
        
        return saved_count
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image for visualization"""
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image
    
    def run_detection(self, save_full_image=True, show_preview=False):
        """Run single detection cycle"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Capture image
        self.logger.info("Capturing image...")
        image = self.capture_image()
        if image is None:
            return False
        
        # Detect stuffed animals
        self.logger.info("Detecting stuffed animals...")
        detections = self.detect_with_yolo(image)
        
        if not detections:
            self.logger.info("No stuffed animals detected")
            return False
        
        self.logger.info(f"Found {len(detections)} stuffed animal(s)")
        
        # Save detections
        saved_count = self.save_detections(image, detections, timestamp)
        
        # Optionally save full image with annotations
        if save_full_image:
            annotated_image = self.draw_detections(image, detections)
            full_image_path = os.path.join(self.output_dir, f"{timestamp}_full_annotated.jpg")
            cv2.imwrite(full_image_path, annotated_image)
            self.logger.info(f"Saved annotated full image: {full_image_path}")
        
        # Optionally show preview
        if show_preview:
            preview_image = self.draw_detections(image, detections)
            cv2.imshow("Detections", cv2.resize(preview_image, (640, 480)))
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()
        
        return True
    
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
    parser = argparse.ArgumentParser(description="Detect stuffed animals using YOLO")
    parser.add_argument("--model-path", default="yolo_models", help="Path to YOLO model files")
    parser.add_argument("--output-dir", default="detected_animals", help="Output directory for detected regions")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--continuous", action="store_true", help="Run continuous detection")
    parser.add_argument("--interval", type=int, default=5, help="Interval between captures (seconds)")
    parser.add_argument("--preview", action="store_true", help="Show detection preview")
    
    args = parser.parse_args()
    
    # Create detector
    detector = StuffedAnimalDetector(
        model_path=args.model_path,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence
    )
    
    try:
        if args.continuous:
            print("Starting continuous detection. Press Ctrl+C to stop...")
            while True:
                detector.run_detection(show_preview=args.preview)
                time.sleep(args.interval)
        else:
            print("Running single detection...")
            detector.run_detection(show_preview=args.preview)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()