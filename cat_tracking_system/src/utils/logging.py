"""
Logging utilities for the cat tracking system
"""
import os
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Dict, Optional


def setup_logging(
    level: str = "INFO",
    log_dir: str = "data/logs",
    console: bool = True,
    file: bool = True,
    max_log_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Setup logging for the cat tracking system
    
    Args:
        level: Logging level
        log_dir: Log directory
        console: Enable console logging
        file: Enable file logging
        max_log_size: Maximum log file size
        backup_count: Number of backup log files
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if file:
        # Create log directory if not exists
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        
        # Create timestamped log file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/cat_tracking_{timestamp}.log"
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Configure other libraries' loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Log startup message
    logger.info(f"Logging initialized at level {level}")


def get_level_color(level: int) -> str:
    """
    Get ANSI color code for log level
    
    Args:
        level: Logging level
        
    Returns:
        ANSI color code
    """
    if level >= logging.CRITICAL:
        return "\033[1;31m"  # Bold Red
    elif level >= logging.ERROR:
        return "\033[31m"    # Red
    elif level >= logging.WARNING:
        return "\033[33m"    # Yellow
    elif level >= logging.INFO:
        return "\033[32m"    # Green
    else:
        return "\033[0m"     # Reset


class ColoredConsoleHandler(logging.StreamHandler):
    """Custom handler for colored console output"""
    
    def emit(self, record):
        # Add color based on log level
        color_code = get_level_color(record.levelno)
        reset_code = "\033[0m"
        
        # Save original formatter
        formatter = self.formatter
        
        # Format with colors
        if formatter:
            self.setFormatter(logging.Formatter(
                f"{color_code}%(levelname)s{reset_code}: %(message)s"
            ))
        
        # Call parent emit
        super().emit(record)
        
        # Restore original formatter
        self.setFormatter(formatter)