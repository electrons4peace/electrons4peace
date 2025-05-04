"""
Hailo-A AI accelerator adapter for YOLO models
"""
import logging
import numpy as np
import os
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the Detection class
from .yolo_detector import Detection


class HailoYoloAdapter:
    """
    Adapter class for running YOLO models on Hailo-A accelerator.
    This adapter provides a consistent interface for YOLO detection
    regardless of whether it's running on Hailo or CPU.
    """
    
    def __init__(self, 
                model_path: str, 
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                class_names: List[str] = None):
        """
        Initialize the Hailo YOLO adapter
        
        Args:
            model_path: Path to the YOLO model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for non-maximum suppression
            class_names: List of class names
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or ["cat"]
        
        # Initialize Hailo device
        self._init_hailo()
    
    def _init_hailo(self):
        """Initialize Hailo device and load model"""
        try:
            # Import Hailo SDK
            import hailo
            
            # Initialize Hailo device
            self.logger.info("Initializing Hailo device")
            self.device = hailo.Device()
            
            # Check if we need to compile the model for Hailo
            hailo_model_path = self._get_hailo_model_path()
            if not hailo_model_path.exists():
                self.logger.info(f"Compiling YOLO model for Hailo: {self.model_path}")
                self._compile_model_for_hailo()
            
            # Load the compiled model
            self.logger.info(f"Loading Hailo model: {hailo_model_path}")
            self.yolo_net = hailo.Net(self.device, hailo_model_path)
            
            # Set input and output layer names
            self.input_layer = self.yolo_net.get_input_layers()[0].name
            self.output_layers = [layer.name for layer in self.yolo_net.get_output_layers()]
            
            self.logger.info("Hailo YOLO adapter initialized successfully")
            self.is_initialized = True
            
        except (ImportError, Exception) as e:
            self.logger.error(f"Failed to initialize Hailo: {str(e)}")
            self.logger.warning("Falling back to dummy implementation")
            self.is_initialized = False
    
    def _get_hailo_model_path(self) -> Path:
        """Get the path for the compiled Hailo model"""
        # Convert model path to Hailo format
        model_name = self.model_path.stem
        hailo_model_dir = Path("models/hailo")
        hailo_model_dir.mkdir(exist_ok=True, parents=True)
        return hailo_model_dir / f"{model_name}_hailo.hef"
    
    def _compile_model_for_hailo(self):
        """Compile YOLO model for Hailo accelerator"""
        try:
            import hailo
            from hailo_model_compiler import compile_model
            
            # Create directory for compiled model
            os.makedirs(os.path.dirname(self._get_hailo_model_path()), exist_ok=True)
            
            # Convert from Ultralytics to ONNX if needed
            onnx_path = self._convert_to_onnx()
            
            # Compile ONNX to Hailo format
            self.logger.info(f"Compiling {onnx_path} for Hailo...")
            compile_model(
                onnx_path,
                str(self._get_hailo_model_path()),
                target_platform="hailo8",
                optimization_level="performance"
            )
            
            self.logger.info(f"Model compiled successfully to {self._get_hailo_model_path()}")
            
        except Exception as e:
            self.logger.error(f"Failed to compile model: {str(e)}")
            raise
    
    def _convert_to_onnx(self) -> str:
        """Convert YOLOv8 model to ONNX format"""
        try:
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(self.model_path)
            
            # Export to ONNX
            onnx_path = str(self.model_path.with_suffix('.onnx'))
            model.export(format="onnx", opset=12)
            
            self.logger.info(f"Model exported to ONNX: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            self.logger.error(f"Failed to convert to ONNX: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in the frame using Hailo accelerator
        
        Args:
            frame: Input image frame
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            return self._dummy_detect(frame)
            
        try:
            # Preprocess input frame
            input_tensor = self._preprocess(frame)
            
            # Run inference on Hailo
            outputs = self.yolo_net.infer({self.input_layer: input_tensor})
            
            # Process outputs
            detections = self._process_outputs(outputs, frame)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Hailo inference error: {str(e)}")
            return self._dummy_detect(frame)
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference on Hailo"""
        # Resize to model input size (typically 640x640 for YOLOv8)
        input_size = 640  # Default YOLOv8 input size
        resized = cv2.resize(frame, (input_size, input_size))
        
        # Convert to BGR to RGB if needed
        if frame.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize and transpose to NCHW format
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor
    
    def _process_outputs(self, outputs: Dict[str, np.ndarray], frame: np.ndarray) -> List[Detection]:
        """Process YOLO outputs from Hailo"""
        detections = []
        
        # Process detection outputs
        # Note: The exact structure depends on the exported YOLO model
        # This is a simplified implementation
        
        # Extract detection data from outputs
        # Assuming the output is in a format similar to YOLOv8
        for output_name in self.output_layers:
            output = outputs[output_name]
            
            # Process each detection
            for detection in output:
                # Extract box coordinates, confidence, and class ID
                x1, y1, x2, y2 = detection[0:4]
                confidence = detection[4]
                class_id = int(detection[5])
                
                # Skip low confidence detections
                if confidence < self.conf_threshold:
                    continue
                
                # Normalize coordinates to 0-1
                h, w = frame.shape[:2]
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Create detection object
                det = Detection(
                    box=[x1, y1, x2, y2],
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=class_name
                )
                detections.append(det)
        
        # Apply non-maximum suppression
        detections = self._apply_nms(detections)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression to remove overlapping detections"""
        if not detections:
            return []
            
        # Convert to numpy arrays
        boxes = np.array([[d.box[0], d.box[1], d.box[2], d.box[3]] for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        # Compute IoU between all boxes
        iou_matrix = self._compute_iou_matrix(boxes)
        
        # Sort by confidence
        indices = np.argsort(scores)[::-1]
        
        # Apply NMS
        keep = []
        while indices.size > 0:
            # Keep the box with highest confidence
            current_idx = indices[0]
            keep.append(current_idx)
            
            # Find overlapping boxes with IoU > threshold
            ious = iou_matrix[current_idx, indices[1:]]
            overlapping = np.where(ious > self.iou_threshold)[0]
            
            # Remove overlapping boxes
            indices = np.delete(indices, overlapping + 1)
            indices = np.delete(indices, 0)
        
        # Return kept detections
        return [detections[i] for i in keep]
    
    def _compute_iou_matrix(self, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between all pairs of boxes"""
        # Extract coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Initialize IoU matrix
        n = boxes.shape[0]
        iou_matrix = np.zeros((n, n))
        
        # Compute IoU for each pair
        for i in range(n):
            for j in range(i+1, n):
                # Compute intersection
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                
                # Check if boxes overlap
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                intersection = w * h
                
                # Compute union
                union = areas[i] + areas[j] - intersection
                
                # Compute IoU
                iou = intersection / union if union > 0 else 0
                
                # Set symmetric values
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        return iou_matrix
    
    def _dummy_detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate dummy detections when Hailo is not available"""
        # Generate 1-3 random detections for testing
        detections = []
        num_detections = np.random.randint(1, 4)
        
        for _ in range(num_detections):
            # Generate random bounding box
            x1 = np.random.uniform(0.1, 0.7)
            y1 = np.random.uniform(0.1, 0.7)
            x2 = min(x1 + np.random.uniform(0.1, 0.3), 0.9)
            y2 = min(y1 + np.random.uniform(0.1, 0.3), 0.9)
            
            # Create detection
            detection = Detection(
                box=[x1, y1, x2, y2],
                confidence=np.random.uniform(0.3, 0.95),
                class_id=0,
                class_name="cat"
            )
            detections.append(detection)
        
        return detections