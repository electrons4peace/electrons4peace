"""
Time manager module for scheduling system operation
"""
import logging
import threading
import time
import datetime
from typing import Dict, List, Callable, Any, Optional


class TimeManager:
    """
    Manages the scheduling of system operation.
    Allows the system to run only at specified times.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the time manager
        
        Args:
            config: Configuration dictionary with schedule settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Default to running all the time if no schedule provided
        self.schedule = config.get("schedule", [])
        
        # Schedule format: List of time ranges
        # [{"start": "08:00", "end": "10:00"}, {"start": "16:00", "end": "18:00"}]
        if not self.schedule:
            self.logger.info("No schedule provided, system will run continuously")
            self.schedule = [{"start": "00:00", "end": "23:59"}]
        
        self.check_interval = config.get("check_interval", 60)  # Check schedule every 60 seconds
        self.running = False
        self.scheduler_thread = None
        self.active = False  # Whether system is currently active based on schedule
        
        # Current callback
        self.callback = None
        self.callback_thread = None
        self.callback_running = False
    
    def start(self, callback: Callable):
        """
        Start the scheduler
        
        Args:
            callback: Function to call when schedule is active
        """
        if self.running:
            self.logger.warning("Scheduler already running")
            return
        
        # Store callback
        self.callback = callback
        
        # Start scheduler thread
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        
        # Stop callback if running
        self._stop_callback()
        
        # Wait for thread to end
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)
        
        self.logger.info("Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Check if current time is within schedule
                should_be_active = self._is_time_active()
                
                # Handle state changes
                if should_be_active and not self.active:
                    self.logger.info("Schedule active, starting system")
                    self._start_callback()
                elif not should_be_active and self.active:
                    self.logger.info("Schedule inactive, stopping system")
                    self._stop_callback()
                
                # Update active state
                self.active = should_be_active
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
            
            # Sleep before next check
            time.sleep(self.check_interval)
    
    def _is_time_active(self) -> bool:
        """
        Check if current time is within schedule
        
        Returns:
            True if system should be active, False otherwise
        """
        # Get current time
        now = datetime.datetime.now().time()
        
        # Check if current time is within any scheduled range
        for time_range in self.schedule:
            start_str = time_range.get("start", "00:00")
            end_str = time_range.get("end", "23:59")
            
            # Parse time strings
            try:
                start_time = datetime.datetime.strptime(start_str, "%H:%M").time()
                end_time = datetime.datetime.strptime(end_str, "%H:%M").time()
                
                # Check if current time is within range
                if start_time <= now <= end_time:
                    return True
            except ValueError as e:
                self.logger.error(f"Invalid time format in schedule: {e}")
        
        return False
    
    def _start_callback(self):
        """Start the callback in a separate thread"""
        if self.callback_running:
            return
        
        # Start callback thread
        self.callback_running = True
        self.callback_thread = threading.Thread(target=self._callback_loop)
        self.callback_thread.daemon = True
        self.callback_thread.start()
    
    def _stop_callback(self):
        """Stop the callback thread"""
        self.callback_running = False
        
        # Wait for thread to end
        if self.callback_thread:
            self.callback_thread.join(timeout=1.0)
            self.callback_thread = None
    
    def _callback_loop(self):
        """Run callback repeatedly while active"""
        while self.callback_running and self.running:
            try:
                # Call the callback
                self.callback()
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in callback: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error