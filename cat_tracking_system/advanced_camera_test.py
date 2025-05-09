#!/usr/bin/env python3
"""
Advanced Raspberry Pi Camera Test for IMX708 Noir Camera
This script tests the camera with multiple modes and provides detailed information
about the camera capabilities.
"""

import time
import os
from datetime import datetime
import argparse
import sys
import json

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 module not found.")
    print("Install with: sudo apt install -y python3-picamera2")
    sys.exit(1)

def save_camera_info(camera, filename="camera_info.json"):
    """Save detailed camera information to a JSON file"""
    info = {
        "model": camera.camera_properties.get("Model", "Unknown"),
        "location": camera.camera_properties.get("Location", "Unknown"),
        "capabilities": camera.camera_properties.get("Capabilities", []),
        "rotation": camera.camera_properties.get("Rotation", 0),
        "raw_formats": camera.camera_properties.get("PixelFormats", []),
    }
    
    # Add available configurations
    configs = []
    for i, config in enumerate(camera.camera_config):
        configs.append({
            "index": i,
            "config": str(config)
        })
    info["configurations"] = configs
    
    # Add controls
    info["controls"] = camera.camera_controls
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(info, f, indent=2)
    
    return filename

def test_camera(capture_mode="still", output_dir="camera_test", delay=2, 
                resolution=None, custom_controls=None, raw=False, 
                save_info=True, burst=False, burst_count=5):
    """
    Test the Raspberry Pi camera with various modes and options
    
    Args:
        capture_mode: "still", "preview", or "raw"
        output_dir: Directory to save output files
        delay: Time to wait before capture to allow camera to adjust
        resolution: Optional tuple (width, height) or None for default
        custom_controls: Dictionary of camera controls to set
        raw: Whether to capture in RAW format (DNG)
        save_info: Whether to save camera information to JSON
        burst: Whether to capture a burst of images
        burst_count: Number of images to capture in burst mode
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Initialize the camera
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Save camera information if requested
    if save_info:
        info_file = os.path.join(output_dir, f"camera_info_{timestamp}.json")
        save_camera_info(picam2, info_file)
        print(f"Camera information saved to {info_file}")
    
    # Configure camera based on the mode
    if capture_mode == "still":
        config = picam2.create_still_configuration()
        output_file = os.path.join(output_dir, f"still_{timestamp}.jpg")
        if raw:
            config["enable_raw"] = True
    elif capture_mode == "preview":
        config = picam2.create_preview_configuration()
        output_file = os.path.join(output_dir, f"preview_{timestamp}.jpg")
    elif capture_mode == "video":
        config = picam2.create_video_configuration()
        output_file = os.path.join(output_dir, f"video_frame_{timestamp}.jpg")
    else:
        print(f"Error: Unknown capture mode '{capture_mode}'")
        sys.exit(1)
    
    # Set custom resolution if provided
    if resolution:
        config["main"]["size"] = resolution
    
    # Apply configuration
    picam2.configure(config)
    
    # Set custom controls if provided
    if custom_controls:
        picam2.set_controls(custom_controls)
    
    # Start the camera
    picam2.start()
    print(f"Camera started with configuration mode: {capture_mode}")
    print(f"Resolution: {config['main']['size']}")
    
    # Allow camera to adjust exposure and white balance
    print(f"Allowing camera to adjust for {delay} seconds...")
    time.sleep(delay)
    
    # Capture image(s)
    if burst:
        print(f"Capturing burst of {burst_count} images...")
        burst_files = []
        for i in range(burst_count):
            burst_file = os.path.join(output_dir, f"burst_{timestamp}_{i}.jpg")
            picam2.capture_file(burst_file)
            burst_files.append(burst_file)
            time.sleep(0.1)  # Small delay between burst captures
        print(f"Burst capture completed. {burst_count} images saved in {output_dir}")
        output_file = burst_files
    else:
        print(f"Capturing image and saving to {output_file}...")
        picam2.capture_file(output_file)
        
        # Capture RAW DNG if requested
        if raw:
            raw_file = os.path.join(output_dir, f"raw_{timestamp}.dng")
            picam2.capture_file(raw_file, formats="raw")
            print(f"RAW image saved to {raw_file}")
            
        print("Image captured successfully!")
    
    # Stop camera
    picam2.stop()
    picam2.close()
    
    print("Camera test completed successfully.")
    return output_file

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Test for Raspberry Pi Camera')
    parser.add_argument('-m', '--mode', type=str, default='still',
                      choices=['still', 'preview', 'video'],
                      help='Capture mode (still, preview, or video)')
    parser.add_argument('-o', '--output', type=str, default='camera_test',
                      help='Output directory for captured images')
    parser.add_argument('-d', '--delay', type=int, default=2,
                      help='Delay in seconds before capture to allow camera to adjust')
    parser.add_argument('-r', '--resolution', type=str,
                      help='Resolution in WxH format (e.g. 4000x3000)')
    parser.add_argument('--raw', action='store_true',
                      help='Capture in RAW format (DNG) in addition to JPEG')
    parser.add_argument('--awb', type=str, default=None, 
                      choices=['auto', 'incandescent', 'tungsten', 'fluorescent', 'indoor', 'daylight', 'cloudy'],
                      help='White balance mode')
    parser.add_argument('--exposure', type=str, default=None,
                      choices=['normal', 'sport', 'short', 'long', 'custom'],
                      help='Exposure mode')
    parser.add_argument('--ev', type=float, default=None,
                      help='Exposure compensation (-10.0 to 10.0)')
    parser.add_argument('--iso', type=int, default=None,
                      help='ISO value (100 to 1600)')
    parser.add_argument('--burst', action='store_true',
                      help='Capture a burst of images')
    parser.add_argument('--burst-count', type=int, default=5,
                      help='Number of images to capture in burst mode')
    parser.add_argument('--no-info', action='store_true',
                      help='Do not save camera information to JSON')
    
    args = parser.parse_args()
    
    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            print(f"Error: Invalid resolution format '{args.resolution}'. Use WxH format (e.g. 4000x3000)")
            sys.exit(1)
    
    # Build custom controls dictionary
    custom_controls = {}
    
    if args.awb:
        custom_controls['AwbMode'] = args.awb
    
    if args.exposure:
        custom_controls['AeMode'] = args.exposure
    
    if args.ev is not None:
        custom_controls['ExposureValue'] = args.ev
    
    if args.iso is not None:
        custom_controls['AnalogueGain'] = args.iso / 100
    
    try:
        output_file = test_camera(
            capture_mode=args.mode,
            output_dir=args.output,
            delay=args.delay,
            resolution=resolution,
            custom_controls=custom_controls if custom_controls else None,
            raw=args.raw,
            save_info=not args.no_info,
            burst=args.burst,
            burst_count=args.burst_count
        )
        
        if isinstance(output_file, list):
            print(f"Test completed successfully. {len(output_file)} images saved in {args.output}")
        else:
            print(f"Test completed successfully. Image saved: {output_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Camera test failed. Please check if the camera is properly connected.")
        sys.exit(1)