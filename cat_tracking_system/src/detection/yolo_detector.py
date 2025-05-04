"""
YOLO-based object detector for cat detection
"""
import os
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Define a class to hold detection results
class Detection:
    def __init__(self, 
                box: List[float], 
                confidence: float, 
                class_id: int, 
                class_name: str):
        """
        Initialize detection object
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2] (normalized 0-1)
            confidence: Detection confidence
            class_id: Class ID
            class_name: Class name
        """
        self.box = box  # [x1, y1, x2, y2] format, normalized 0-1
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name


class YoloDetector:
    """
    YOLO object detector using YOLOv8.
    Compatible with Hailo-A accelerator for Raspberry Pi.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the YOLO detector
        
        Args:
            config: Detection configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Detector settings
        self.model_path = Path(config.get("model_path", "models/yolo/yolov8n.pt"))
        self.conf_threshold = config.get("confidence_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.device = config.get("device", "cpu")
        
        # Class names
        self.class_names = config.get("class_names", ["cat"])
        
        # Use Hailo acceleration if available
        self.use_hailo = config.get("use_hailo", False)
        
        # Initialize detector
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the YOLO detector"""
        try:
            # Check if the model file exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}")
                self.logger.info("Attempting to download YOLOv8n model...")
                self._download_model()
            
            # Import and load model based on configuration
            if self.use_hailo:
                self._init_hailo_detector()
            else:
                self._init_ultralytics_detector()
            
            self.logger.info(f"YOLO detector initialized with model: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize detector: {str(e)}")
            # Initialize a dummy detector for testing
            self.model = None
            self.logger.warning("Using dummy detector for testing")
    
    def _init_ultralytics_detector(self):
        """Initialize detector using Ultralytics YOLOv8"""
        try:
            # Import inside function to handle case where package isn't installed
            from ultralytics import YOLO
            
            # Load the model
            self.model = YOLO(self.model_path)
            self.is_dummy = False
            self.logger.info("Initialized Ultralytics YOLOv8 detector")
        except ImportError:
            self.logger.error("Failed to import ultralytics. Is it installed?")
            self.logger.warning("Using dummy detector")
            self.model = None
            self.is_dummy = True
    
    def _init_hailo_detector(self):
        """Initialize detector using Hailo-A accelerator"""
        try:
            # Import hailo_adapter module
            from .hailo_adapter import HailoYoloAdapter
            
            # Initialize Hailo adapter
            self.model = HailoYoloAdapter(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                class_names=self.class_names
            )
            self.is_dummy = False
            self.logger.info("Initialized Hailo YOLO detector")
        except ImportError:
            self.logger.error("Failed to import hailo_adapter. Falling back to CPU.")
            self._init_ultralytics_detector()
    
    def _download_model(self):
        """Download YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Download nano model
            model = YOLO("yolov8n.pt")
            
            # Save model to specified path
            model.save(self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to download model: {str(e)}")
            raise
    
    def detect(self, frames: Dict[str, np.ndarray]) -> List[Detection]:
        """
        Detect objects in the frames
        
        Args:
            frames: Dictionary with camera names as keys and frames as values
            
        Returns:
            List of Detection objects
        """
        if not frames:
            return []
        
        # If dummy detector, return dummy detections
        if self.model is None or self.is_dummy:
            return self._dummy_detect(frames)
        
        detections = []
        
        # Use visible camera for detection if available
        if "visible" in frames:
            frame = frames["visible"]
            
            try:
                # Run detection using YOLOv8
                if self.use_hailo:
                    # Hailo adapter has a different interface
                    results = self.model.detect(frame)
                else:
                    # Ultralytics YOLO
                    results = self.model(
                        frame, 
                        conf=self.conf_threshold, 
                        iou=self.iou_threshold,
                        device=self.device
                    )
                
                # Process detection results
                detections = self._process_results(results, frame)
            except Exception as e:
                self.logger.error(f"Detection error: {str(e)}")
                return []
        
        # NIR camera can be used to verify or enhance detections
        # This simple implementation doesn't use NIR data yet
        
        return detections
    
    def _process_results(self, results, frame) -> List[Detection]:
        """Process detection results from YOLOv8"""
        detections = []
        
        # Handle Ultralytics YOLO results
        if not self.use_hailo:
            try:
                # Get the first result (single image)
                result = results[0]
                
                # Extract bounding boxes, confidences, and class IDs
                boxes = result.boxes.xyxyn.cpu().numpy()  # normalized coordinates
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Create Detection objects
                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    # Get class name
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    # Create detection
                    detection = Detection(
                        box=box.tolist(),
                        confidence=float(confidence),
                        class_id=int(class_id),
                        class_name=class_name
                    )
                    detections.append(detection)
            except Exception as e:
                self.logger.error(f"Error processing results: {str(e)}")
        else:
            # Hailo adapter already returns Detection objects
            detections = results
        
        return detections
    
    def _dummy_detect(self, frames: Dict[str, np.ndarray]) -> List[Detection]:
        """Generate dummy detections for testing"""
        detections = []
        
        if "visible" in frames:
            frame = frames["visible"]
            h, w = frame.shape[:2]
            
            # Generate 1-3 random "cat" detections
            num_detections = np.random.randint(1, 4)
            
            for _ in range(num_detections):
                # Generate random box
                x1 = np.random.uniform(0.1, 0.7)
                y1 = np.random.uniform(0.1, 0.7)
                x2 = min(x1 + np.random.uniform(0.1, 0.3), 0.9)
                y2 = min(y1 + np.random.uniform(0.1, 0.3), 0.9)
                
                # Create detection with random confidence
                detection = Detection(
                    box=[x1, y1, x2, y2],
                    confidence=np.random.uniform(0.3, 0.95),
                    class_id=0,
                    class_name="cat"
                )
                detections.append(detection)
        
        return detections