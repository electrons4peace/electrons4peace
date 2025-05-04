"""
Object detection module
"""
from .yolo_detector import YoloDetector, Detection
from .hailo_adapter import HailoYoloAdapter

__all__ = ['YoloDetector', 'Detection', 'HailoYoloAdapter']