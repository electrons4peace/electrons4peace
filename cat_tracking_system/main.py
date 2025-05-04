#!/usr/bin/env python3
"""
Cat Tracking System - Main Application Entry Point
"""
import os
import time
import yaml
import argparse
import logging
from pathlib import Path

from src.utils.logging import setup_logging
from src.acquisition.camera_manager import CameraManager
from src.detection.yolo_detector import YoloDetector
from src.tracking.simple_tracker import SimpleTracker
from src.identification.cat_profiles import CatProfileManager
from src.dashboard.web_server import WebDashboard
from src.scheduler.time_manager import TimeManager
from src.auto_annotation.data_collector import DataCollector


class CatTrackingSystem:
    """Main application class for the Cat Tracking System"""
    
    def __init__(self, config_path="config/system.yaml"):
        """Initialize the cat tracking system"""
        # Load configuration
        self.config_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "config"
        self.system_config = self._load_config(config_path)
        
        # Setup logging
        setup_logging(self.system_config.get("logging", {}).get("level", "INFO"))
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Cat Tracking System")
        
        # Initialize components
        self._init_components()
        
        # System state
        self.running = False
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        config_file = self.config_dir / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_components(self):
        """Initialize system components"""
        # Load component-specific configurations
        camera_config = self._load_config("cameras.yaml")
        detection_config = self._load_config("detection.yaml")
        identification_config = self._load_config("identification.yaml")
        
        # Initialize components
        self.camera_manager = CameraManager(camera_config)
        self.detector = YoloDetector(detection_config)
        self.tracker = SimpleTracker()
        self.profile_manager = CatProfileManager(identification_config)
        self.data_collector = DataCollector(
            self.system_config.get("data_collection", {})
        )
        
        # Initialize dashboard if enabled
        if self.system_config.get("dashboard", {}).get("enabled", True):
            dashboard_port = self.system_config.get("dashboard", {}).get("port", 8080)
            self.dashboard = WebDashboard(port=dashboard_port)
        else:
            self.dashboard = None
        
        # Initialize scheduler if system should run at specific times
        schedule_config = self.system_config.get("schedule", {})
        if schedule_config.get("enabled", True):
            self.time_manager = TimeManager(schedule_config)
        else:
            self.time_manager = None
    
    def start(self):
        """Start the cat tracking system"""
        self.logger.info("Starting Cat Tracking System")
        self.running = True
        
        # Start the dashboard if enabled
        if self.dashboard:
            self.dashboard.start()
            self.logger.info(f"Dashboard started on port {self.dashboard.port}")
        
        # Start the scheduler if enabled
        if self.time_manager:
            self.time_manager.start(callback=self.run_detection_cycle)
            self.logger.info("Scheduler started")
            # Let the scheduler handle the detection cycles
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
        else:
            # Run continuously
            self.logger.info("Running in continuous mode")
            self.run_continuous()
    
    def run_continuous(self):
        """Run the system in continuous mode"""
        try:
            while self.running:
                self.run_detection_cycle()
        except KeyboardInterrupt:
            self.stop()
    
    def run_detection_cycle(self):
        """Run a single detection cycle"""
        try:
            # Capture frames from both cameras
            frames = self.camera_manager.capture_frames()
            if not frames:
                self.logger.warning("No frames captured")
                return
            
            # Run detection on frames
            detections = self.detector.detect(frames)
            
            # Update tracking
            tracked_objects = self.tracker.update(detections)
            
            # Identify cats from tracked objects
            identified_cats = self.profile_manager.identify(tracked_objects, frames)
            
            # Collect data for training if auto-annotation is enabled
            if self.system_config.get("data_collection", {}).get("enabled", True):
                self.data_collector.collect(frames, identified_cats)
            
            # Update dashboard with latest results
            if self.dashboard:
                self.dashboard.update_data(identified_cats)
            
            # Log results
            self.logger.info(f"Detected {len(identified_cats)} cats")
            
        except Exception as e:
            self.logger.error(f"Error in detection cycle: {str(e)}")
    
    def stop(self):
        """Stop the cat tracking system"""
        self.logger.info("Stopping Cat Tracking System")
        self.running = False
        
        # Stop components
        if self.camera_manager:
            self.camera_manager.close()
        
        if self.dashboard:
            self.dashboard.stop()
        
        if self.time_manager:
            self.time_manager.stop()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cat Tracking System")
    parser.add_argument(
        "--config", 
        type=str, 
        default="system.yaml",
        help="Path to system configuration file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cat_system = CatTrackingSystem(config_path=args.config)
    cat_system.start()