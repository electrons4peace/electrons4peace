"""
Camera manager module for handling dual camera setup (visible and NIR)
"""
import cv2
import logging
import threading
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CameraManager:
    """
    Manages multiple cameras for the cat tracking system.
    Handles synchronization between visible and NIR cameras.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the camera manager
        
        Args:
            config: Camera configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.cameras = {}
        self.frame_buffers = {}
        self.running = False
        self.capture_thread = None
        
        # Camera-specific configurations
        self.visible_config = config.get("visible", {})
        self.nir_config = config.get("nir", {})
        
        # Frame synchronization settings
        self.sync_threshold_ms = config.get("sync", {}).get("threshold_ms", 50)
        
        # Initialize cameras
        self._init_cameras()
    
    def _init_cameras(self):
        """Initialize and configure the cameras"""
        # Initialize visible camera
        visible_id = self.visible_config.get("id", 0)
        try:
            self.cameras["visible"] = self._setup_camera(
                visible_id,
                self.visible_config.get("width", 640),
                self.visible_config.get("height", 480),
                self.visible_config.get("fps", 30)
            )
            self.logger.info(f"Visible camera initialized (id: {visible_id})")
        except Exception as e:
            self.logger.error(f"Failed to initialize visible camera: {str(e)}")
            raise
        
        # Initialize NIR camera
        nir_id = self.nir_config.get("id", 1)
        try:
            self.cameras["nir"] = self._setup_camera(
                nir_id,
                self.nir_config.get("width", 640),
                self.nir_config.get("height", 480),
                self.nir_config.get("fps", 30)
            )
            self.logger.info(f"NIR camera initialized (id: {nir_id})")
        except Exception as e:
            self.logger.error(f"Failed to initialize NIR camera: {str(e)}")
            # If NIR camera initialization fails, we can still continue with just visible
            self.logger.warning("Continuing with visible camera only")
        
        # Initialize frame buffers
        for camera_name in self.cameras:
            self.frame_buffers[camera_name] = []
    
    def _setup_camera(self, camera_id, width, height, fps):
        """Set up a camera with the given parameters"""
        # For testing on systems without cameras, use a dummy camera
        if self.config.get("use_dummy", False):
            self.logger.info(f"Using dummy camera for {camera_id}")
            return DummyCamera(camera_id, width, height)
        
        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            raise ValueError(f"Could not open camera with id {camera_id}")
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, fps)
        
        return camera
    
    def start_capture(self):
        """Start continuous frame capture in a separate thread"""
        if self.running:
            return
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.logger.info("Camera capture thread started")
    
    def _capture_loop(self):
        """Continuously capture frames from all cameras"""
        while self.running:
            for camera_name, camera in self.cameras.items():
                success, frame = camera.read()
                if success:
                    timestamp = time.time()
                    # Add frame to buffer with timestamp
                    self.frame_buffers[camera_name].append((frame, timestamp))
                    
                    # Keep buffer size limited
                    max_buffer_size = self.config.get("buffer_size", 10)
                    if len(self.frame_buffers[camera_name]) > max_buffer_size:
                        self.frame_buffers[camera_name].pop(0)
                else:
                    self.logger.warning(f"Failed to capture frame from {camera_name} camera")
            
            # Control capture rate
            time.sleep(1.0 / self.config.get("internal_fps", 30))
    
    def capture_frames(self) -> Dict[str, np.ndarray]:
        """
        Capture synchronized frames from all cameras
        
        Returns:
            Dictionary with camera names as keys and frames as values
        """
        # Start capture thread if not already running
        if not self.running:
            self.start_capture()
            # Wait for buffers to fill
            time.sleep(0.5)
        
        # If only one camera is available, just return its latest frame
        if len(self.cameras) == 1:
            camera_name = list(self.cameras.keys())[0]
            if self.frame_buffers[camera_name]:
                frame, _ = self.frame_buffers[camera_name][-1]
                return {camera_name: frame}
            return {}
        
        # Get synchronized frames from buffers
        return self._get_synchronized_frames()
    
    def _get_synchronized_frames(self) -> Dict[str, np.ndarray]:
        """Get synchronized frames from the buffers"""
        # Check if we have frames in all buffers
        for camera_name, buffer in self.frame_buffers.items():
            if not buffer:
                self.logger.warning(f"No frames in buffer for {camera_name}")
                return {}
        
        # Find the closest frames by timestamp
        result = {}
        
        # Use the latest frame from visible camera as reference
        if "visible" in self.frame_buffers and self.frame_buffers["visible"]:
            reference_frame, reference_timestamp = self.frame_buffers["visible"][-1]
            result["visible"] = reference_frame
            
            # Find the closest NIR frame
            if "nir" in self.frame_buffers and self.frame_buffers["nir"]:
                closest_frame = None
                min_diff = float('inf')
                
                for frame, timestamp in self.frame_buffers["nir"]:
                    time_diff = abs(timestamp - reference_timestamp) * 1000  # Convert to ms
                    if time_diff < min_diff:
                        min_diff = time_diff
                        closest_frame = frame
                
                # Only use the frame if it's within sync threshold
                if min_diff <= self.sync_threshold_ms:
                    result["nir"] = closest_frame
                    self.logger.debug(f"Frames synchronized with {min_diff:.2f}ms difference")
                else:
                    self.logger.warning(
                        f"Frames not synchronized. Time difference: {min_diff:.2f}ms"
                    )
        
        return result
    
    def close(self):
        """Release all camera resources"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        for camera_name, camera in self.cameras.items():
            if not isinstance(camera, DummyCamera):
                camera.release()
        
        self.cameras = {}
        self.frame_buffers = {}
        self.logger.info("Camera resources released")


class DummyCamera:
    """
    Dummy camera class for testing without physical cameras
    Generates colored frames with timestamp
    """
    
    def __init__(self, camera_id, width, height):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.frame_count = 0
        
        # Create a different color for each dummy camera
        if camera_id == 0:  # visible
            self.color = (0, 255, 0)  # Green
        else:  # NIR
            self.color = (128, 128, 128)  # Gray
    
    def read(self):
        """Read a frame from the dummy camera"""
        # Create a blank frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add a timestamp and frame number
        timestamp = time.strftime("%H:%M:%S")
        self.frame_count += 1
        
        # Add colored rectangle
        cv2.rectangle(frame, (50, 50), (self.width-50, self.height-50), self.color, -1)
        
        # Add text
        cv2.putText(
            frame, 
            f"Camera {self.camera_id} - {timestamp}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        cv2.putText(
            frame, 
            f"Frame: {self.frame_count}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return True, frame
    
    def isOpened(self):
        """Check if dummy camera is opened"""
        return True