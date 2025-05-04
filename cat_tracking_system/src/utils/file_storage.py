"""
File storage utilities for the cat tracking system
"""
import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


class FileStorage:
    """
    Simple file-based storage for cat tracking data.
    Handles saving and loading data in JSON format.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the file storage
        
        Args:
            base_dir: Base directory for storage
        """
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path(base_dir)
        
        # Ensure base directory exists
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self.detection_dir = self.base_dir / "detections"
        self.detection_dir.mkdir(exist_ok=True)
        
        self.config_dir = self.base_dir / "config"
        self.config_dir.mkdir(exist_ok=True)
    
    def save_json(self, data: Dict, filepath: Union[str, Path]) -> bool:
        """
        Save data as JSON
        
        Args:
            data: Data to save
            filepath: Path to save to
            
        Returns:
            Success flag
        """
        try:
            # Convert path to Path object
            if isinstance(filepath, str):
                filepath = Path(filepath)
            
            # Create directory if not exists
            filepath.parent.mkdir(exist_ok=True, parents=True)
            
            # Write data
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON: {str(e)}")
            return False
    
    def load_json(self, filepath: Union[str, Path]) -> Optional[Dict]:
        """
        Load data from JSON
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded data or None if error
        """
        try:
            # Convert path to Path object
            if isinstance(filepath, str):
                filepath = Path(filepath)
            
            # Check if file exists
            if not filepath.exists():
                self.logger.warning(f"File not found: {filepath}")
                return None
            
            # Read data
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON: {str(e)}")
            return None
    
    def save_detection_log(self, detection_data: Dict) -> str:
        """
        Save detection data to a timestamped log file
        
        Args:
            detection_data: Detection data to log
            
        Returns:
            Path to saved file
        """
        # Create timestamp
        timestamp = int(time.time())
        date_str = time.strftime("%Y%m%d", time.localtime(timestamp))
        
        # Create directory for this day if not exists
        day_dir = self.detection_dir / date_str
        day_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = f"detection_{timestamp}.json"
        filepath = day_dir / filename
        
        # Save data
        if self.save_json(detection_data, filepath):
            return str(filepath)
        else:
            return ""
    
    def list_detection_logs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
        """
        List detection log files within date range
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            
        Returns:
            List of log file paths
        """
        # Default to all dates if not specified
        if not start_date:
            start_date = "00000000"
        if not end_date:
            end_date = "99999999"
        
        # Find all day directories
        day_dirs = [d for d in self.detection_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        
        # Filter by date range
        day_dirs = [d for d in day_dirs if start_date <= d.name <= end_date]
        
        # Get all log files
        log_files = []
        for day_dir in day_dirs:
            log_files.extend([str(f) for f in day_dir.glob("*.json")])
        
        return sorted(log_files)
    
    def get_detections_by_date(self, date: str) -> List[Dict]:
        """
        Get all detections for a specific date
        
        Args:
            date: Date string (YYYYMMDD)
            
        Returns:
            List of detection data
        """
        # Get day directory
        day_dir = self.detection_dir / date
        if not day_dir.exists():
            return []
        
        # Load all log files
        detections = []
        for log_file in day_dir.glob("*.json"):
            data = self.load_json(log_file)
            if data:
                detections.append(data)
        
        return detections
    
    def save_config(self, config: Dict, name: str) -> bool:
        """
        Save configuration
        
        Args:
            config: Configuration data
            name: Configuration name
            
        Returns:
            Success flag
        """
        # Ensure config has .yaml extension
        if not name.endswith((".yaml", ".yml")):
            name = f"{name}.yaml"
        
        filepath = self.config_dir / name
        return self.save_json(config, filepath)
    
    def load_config(self, name: str) -> Optional[Dict]:
        """
        Load configuration
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration data or None if error
        """
        # Ensure config has .yaml extension
        if not name.endswith((".yaml", ".yml")):
            name = f"{name}.yaml"
        
        filepath = self.config_dir / name
        return self.load_json(filepath)
    
    def list_configs(self) -> List[str]:
        """
        List available configurations
        
        Returns:
            List of configuration names
        """
        return [f.name for f in self.config_dir.glob("*.yaml")]