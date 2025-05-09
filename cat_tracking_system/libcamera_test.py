#!/usr/bin/env python3
"""
Raspberry Pi Camera Test using libcamera API
This script tests the Raspberry Pi camera using the libcamera Python bindings.
"""

import time
from datetime import datetime
import argparse
import sys

try:
    from picamera2 import Picamera2, Preview
except ImportError:
    print("Error: picamera2 module not found.")
    print("Install with: sudo apt install -y python3-picamera2")
    sys.exit(1)

def test_camera(preview_time=5, resolution=None, output_file=None, info=False):
    """
    Test the Raspberry Pi camera using libcamera via picamera2
    
    Args:
        preview_time: Time in seconds to show the preview (0 for no preview)
        resolution: Tuple of (width, height) or None for default
        output_file: Filename to save the captured image, or None to auto-generate
        info: Whether to display detailed camera information
    """
    # Generate a filename with timestamp if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"camera_test_{timestamp}.jpg"
    
    # Initialize the camera
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Display camera info if requested
    if info:
        print("\nCamera Information:")
        print(f"Camera model: {picam2.camera_properties.get('Model', 'Unknown')}")
        
        # List available camera modes
        print("\nAvailable camera configurations:")
        for i, config in enumerate(picam2.camera_config):
            print(f"  Config {i}: {config}")
            
        # Show available controls
        print("\nAvailable camera controls:")
        for control, value in picam2.camera_controls.items():
            print(f"  {control}: {value}")
    
    # Configure camera
    config = picam2.create_still_configuration()
    if resolution:
        config["main"]["size"] = resolution
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    print(f"Camera started with resolution: {config['main']['size']}")
    
    # Capture image without preview (to avoid event loop issues)
    print(f"Capturing image and saving to {output_file}...")
    
    # Allow camera to settle with auto exposure and white balance
    if preview_time > 0:
        print(f"Allowing camera to adjust for {preview_time} seconds...")
        time.sleep(preview_time)
    
    # Capture the image
    picam2.capture_file(output_file)
    print("Image captured successfully!")
    
    # Stop camera
    picam2.stop()
    picam2.close()
    
    print("Camera test completed.")
    return output_file

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Raspberry Pi Camera using libcamera')
    parser.add_argument('-t', '--time', type=int, default=5,
                      help='Preview time in seconds (0 for no preview)')
    parser.add_argument('-r', '--resolution', type=str,
                      help='Resolution in WxH format (e.g. 1920x1080)')
    parser.add_argument('-o', '--output', type=str,
                      help='Output filename for captured image')
    parser.add_argument('-i', '--info', action='store_true',
                      help='Display detailed camera information')
    
    args = parser.parse_args()
    
    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            print(f"Error: Invalid resolution format '{args.resolution}'. Use WxH format (e.g. 1920x1080)")
            sys.exit(1)
    
    try:
        output_file = test_camera(
            preview_time=args.time,
            resolution=resolution,
            output_file=args.output,
            info=args.info
        )
        print(f"Test completed successfully. Image saved: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        print("Camera test failed. Please check if the camera is properly connected.")
        sys.exit(1)