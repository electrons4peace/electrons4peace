#!/usr/bin/env python3
"""
Dual Camera Background Capture for Raspberry Pi 5
Captures images from two Pi Camera modules (IMX708 NoIR and standard camera)
at 5 fps for background subtraction in stuffed animal detection
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse
import logging
from threading import Thread, Event
from queue import Queue
from picamera2 import Picamera2

class DualCameraBackgroundCapture:
    def __init__(self, output_base_dir="background_images", fps=5, capture_duration=10):
        """
        Initialize dual camera capture system
        
        Args:
            output_base_dir: Base directory for saving images
            fps: Frames per second for each camera
            capture_duration: Total capture duration in seconds
        """
        self.output_base_dir = output_base_dir
        self.fps = fps
        self.capture_duration = capture_duration
        self.frame_interval = 1.0 / fps
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.camera1_dir = os.path.join(output_base_dir, "camera1_noir")
        self.camera2_dir = os.path.join(output_base_dir, "camera2_standard")
        os.makedirs(self.camera1_dir, exist_ok=True)
        os.makedirs(self.camera2_dir, exist_ok=True)
        
        # Camera objects
        self.camera1 = None  # NoIR camera
        self.camera2 = None  # Standard camera
        
        # Threading components
        self.stop_event = Event()
        self.capture_queues = {
            'camera1': Queue(maxsize=30),
            'camera2': Queue(maxsize=30)
        }
        
        # Statistics
        self.stats = {
            'camera1': {'captured': 0, 'saved': 0, 'errors': 0},
            'camera2': {'captured': 0, 'saved': 0, 'errors': 0}
        }
        
    def init_cameras(self):
        """Initialize both Pi Camera modules"""
        try:
            # Initialize NoIR camera (typically on port 0)
            self.logger.info("Initializing Camera 1 (IMX708 NoIR)...")
            self.camera1 = Picamera2(0)
            config1 = self.camera1.create_still_configuration(
                main={"size": (1920, 1080)},  # Full HD for good background detail
                buffer_count=2  # Reduce buffer for faster capture
            )
            self.camera1.configure(config1)
            self.camera1.start()
            time.sleep(2)  # Allow camera to stabilize
            self.logger.info("Camera 1 initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Camera 1: {e}")
            self.camera1 = None
        
        try:
            # Initialize standard camera (typically on port 1)
            self.logger.info("Initializing Camera 2 (Standard)...")
            self.camera2 = Picamera2(1)
            config2 = self.camera2.create_still_configuration(
                main={"size": (1920, 1080)},  # Full HD for good background detail
                buffer_count=2
            )
            self.camera2.configure(config2)
            self.camera2.start()
            time.sleep(2)  # Allow camera to stabilize
            self.logger.info("Camera 2 initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Camera 2: {e}")
            self.camera2 = None
        
        # Check if at least one camera is available
        if self.camera1 is None and self.camera2 is None:
            raise Exception("No cameras could be initialized!")
        
    def capture_frame(self, camera, camera_name):
        """Capture a single frame from specified camera"""
        try:
            if camera is None:
                return None
            
            # Capture frame
            frame = camera.capture_array()
            
            # Convert RGB to BGR for OpenCV if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            self.stats[camera_name]['captured'] += 1
            return frame
            
        except Exception as e:
            self.logger.error(f"Error capturing from {camera_name}: {e}")
            self.stats[camera_name]['errors'] += 1
            return None
    
    def capture_thread(self, camera, camera_name):
        """Thread function for continuous capture from a camera"""
        self.logger.info(f"Starting capture thread for {camera_name}")
        next_capture_time = time.time()
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # Check if it's time to capture
            if current_time >= next_capture_time:
                frame = self.capture_frame(camera, camera_name)
                
                if frame is not None:
                    # Add timestamp to frame data
                    frame_data = {
                        'frame': frame,
                        'timestamp': datetime.now(),
                        'camera': camera_name
                    }
                    
                    # Try to add to queue (non-blocking)
                    try:
                        self.capture_queues[camera_name].put_nowait(frame_data)
                    except:
                        self.logger.warning(f"Queue full for {camera_name}, dropping frame")
                
                # Schedule next capture
                next_capture_time += self.frame_interval
                
                # Handle drift - if we're running behind, reset timing
                if next_capture_time < current_time:
                    next_capture_time = current_time + self.frame_interval
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
        
        self.logger.info(f"Capture thread for {camera_name} stopped")
    
    def save_thread(self):
        """Thread function for saving captured frames"""
        self.logger.info("Starting save thread")
        
        while not self.stop_event.is_set() or any(not q.empty() for q in self.capture_queues.values()):
            # Process frames from both camera queues
            for camera_name, queue in self.capture_queues.items():
                if not queue.empty():
                    try:
                        frame_data = queue.get(timeout=0.1)
                        self.save_frame(frame_data)
                    except:
                        pass
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
        
        self.logger.info("Save thread stopped")
    
    def save_frame(self, frame_data):
        """Save a single frame to disk"""
        try:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            camera_name = frame_data['camera']
            
            # Determine output directory
            output_dir = self.camera1_dir if camera_name == 'camera1' else self.camera2_dir
            
            # Create filename with timestamp
            filename = f"{camera_name}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame with high quality for background subtraction
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.stats[camera_name]['saved'] += 1
            
            # Log progress every 10 saves
            if self.stats[camera_name]['saved'] % 10 == 0:
                self.logger.info(f"{camera_name}: Saved {self.stats[camera_name]['saved']} frames")
            
        except Exception as e:
            self.logger.error(f"Error saving frame from {camera_name}: {e}")
            self.stats[camera_name]['errors'] += 1
    
    def run_capture(self):
        """Main capture function"""
        self.logger.info(f"Starting dual camera capture for {self.capture_duration} seconds at {self.fps} fps")
        self.logger.info(f"Output directories: {self.camera1_dir}, {self.camera2_dir}")
        
        # Initialize cameras
        self.init_cameras()
        
        # Start save thread
        save_thread = Thread(target=self.save_thread, name="SaveThread")
        save_thread.start()
        
        # Start capture threads for available cameras
        capture_threads = []
        
        if self.camera1 is not None:
            thread1 = Thread(target=self.capture_thread, args=(self.camera1, 'camera1'), name="Camera1Thread")
            thread1.start()
            capture_threads.append(thread1)
        
        if self.camera2 is not None:
            thread2 = Thread(target=self.capture_thread, args=(self.camera2, 'camera2'), name="Camera2Thread")
            thread2.start()
            capture_threads.append(thread2)
        
        # Wait for capture duration
        start_time = time.time()
        try:
            while time.time() - start_time < self.capture_duration:
                elapsed = time.time() - start_time
                remaining = self.capture_duration - elapsed
                
                # Print progress
                print(f"\rCapturing... {elapsed:.1f}s / {self.capture_duration}s "
                      f"(Camera1: {self.stats['camera1']['captured']} captured, "
                      f"Camera2: {self.stats['camera2']['captured']} captured)", end='', flush=True)
                
                time.sleep(0.1)
            
            print("\nCapture period complete, finishing saves...")
            
        except KeyboardInterrupt:
            print("\nCapture interrupted by user")
        
        # Stop all threads
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in capture_threads:
            thread.join(timeout=2)
        save_thread.join(timeout=5)
        
        # Final statistics
        self.print_statistics()
        
        # Cleanup
        self.cleanup()
    
    def print_statistics(self):
        """Print capture statistics"""
        print("\n" + "="*50)
        print("CAPTURE STATISTICS")
        print("="*50)
        
        total_captured = 0
        total_saved = 0
        total_errors = 0
        
        for camera_name, stats in self.stats.items():
            print(f"\n{camera_name.upper()}:")
            print(f"  Frames captured: {stats['captured']}")
            print(f"  Frames saved: {stats['saved']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Capture rate: {stats['captured']/self.capture_duration:.2f} fps")
            
            total_captured += stats['captured']
            total_saved += stats['saved']
            total_errors += stats['errors']
        
        print(f"\nTOTAL:")
        print(f"  Total frames captured: {total_captured}")
        print(f"  Total frames saved: {total_saved}")
        print(f"  Total errors: {total_errors}")
        print(f"  Overall capture rate: {total_captured/self.capture_duration:.2f} fps")
        
        print(f"\nImages saved to:")
        print(f"  Camera 1 (NoIR): {self.camera1_dir}")
        print(f"  Camera 2 (Standard): {self.camera2_dir}")
    
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if self.camera1:
                self.camera1.stop()
                self.logger.info("Camera 1 stopped")
            
            if self.camera2:
                self.camera2.stop()
                self.logger.info("Camera 2 stopped")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Capture background images from dual Pi cameras")
    parser.add_argument("--duration", type=int, default=10, 
                       help="Capture duration in seconds (default: 10)")
    parser.add_argument("--fps", type=int, default=5, 
                       help="Frames per second for each camera (default: 5)")
    parser.add_argument("--output-dir", default="background_images", 
                       help="Base output directory (default: background_images)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration < 5 or args.duration > 30:
        print("Duration should be between 5 and 30 seconds")
        return
    
    if args.fps < 1 or args.fps > 10:
        print("FPS should be between 1 and 10")
        return
    
    # Create and run capture system
    capture_system = DualCameraBackgroundCapture(
        output_base_dir=args.output_dir,
        fps=args.fps,
        capture_duration=args.duration
    )
    
    print(f"Dual Camera Background Capture System")
    print(f"Duration: {args.duration} seconds")
    print(f"FPS per camera: {args.fps}")
    print(f"Expected frames per camera: {args.duration * args.fps}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    capture_system.run_capture()

if __name__ == "__main__":
    main()