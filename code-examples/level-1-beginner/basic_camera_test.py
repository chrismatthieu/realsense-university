#!/usr/bin/env python3
"""
Basic RealSense Camera Test
===========================

This script demonstrates basic RealSense camera functionality including:
- Camera initialization and configuration
- Frame capture and display
- Basic error handling
- Camera information display

Author: RealSense University
Level: 1 (Beginner)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import time

class RealSenseCamera:
    """Basic RealSense camera handler"""
    
    def __init__(self, width=640, height=480, fps=30):
        """
        Initialize RealSense camera
        
        Args:
            width (int): Image width
            height (int): Image height
            fps (int): Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.is_streaming = False
        
    def initialize(self):
        """
        Initialize the camera pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(
                rs.stream.depth, 
                self.width, 
                self.height, 
                rs.format.z16, 
                self.fps
            )
            self.config.enable_stream(
                rs.stream.color, 
                self.width, 
                self.height, 
                rs.format.bgr8, 
                self.fps
            )
            
            # Start streaming
            self.pipeline.start(self.config)
            self.is_streaming = True
            
            print("✅ Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def get_frames(self):
        """
        Get current frames from camera
        
        Returns:
            tuple: (depth_image, color_image) or (None, None) if failed
        """
        if not self.is_streaming:
            print("❌ Camera not streaming")
            return None, None
            
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Get individual frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                return depth_image, color_image
            else:
                print("❌ Failed to get frames")
                return None, None
                
        except Exception as e:
            print(f"❌ Error getting frames: {e}")
            return None, None
    
    def get_camera_info(self):
        """
        Get camera information
        
        Returns:
            dict: Camera information or None if failed
        """
        if not self.is_streaming:
            return None
            
        try:
            # Get device info
            device = self.pipeline.get_active_profile().get_device()
            return {
                'name': device.get_info(rs.camera_info.name),
                'serial': device.get_info(rs.camera_info.serial_number),
                'firmware': device.get_info(rs.camera_info.firmware_version)
            }
        except Exception as e:
            print(f"❌ Error getting camera info: {e}")
            return None
    
    def stop(self):
        """Stop camera streaming"""
        if self.pipeline and self.is_streaming:
            self.pipeline.stop()
            self.is_streaming = False
            print("✅ Camera stopped")
    
    def process_depth(self, depth_image):
        """
        Process depth image for visualization
        
        Args:
            depth_image (np.ndarray): Raw depth image
            
        Returns:
            np.ndarray: Processed depth image
        """
        # Normalize depth image for display
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return depth_colormap

def main():
    """Main function to test camera functionality"""
    print("🎥 RealSense Camera Test")
    print("=" * 30)
    
    # Initialize camera
    camera = RealSenseCamera()
    
    if not camera.initialize():
        print("❌ Failed to initialize camera. Exiting.")
        sys.exit(1)
    
    # Get camera information
    camera_info = camera.get_camera_info()
    if camera_info:
        print(f"📷 Camera: {camera_info['name']}")
        print(f"🔢 Serial: {camera_info['serial']}")
        print(f"⚙️ Firmware: {camera_info['firmware']}")
    
    print("\n🎯 Camera test started. Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frames
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                print("❌ Failed to get frames")
                break
            
            # Process depth image
            depth_colormap = camera.process_depth(depth_image)
            
            # Display images
            cv2.imshow('Color Stream', color_image)
            cv2.imshow('Depth Stream', depth_colormap)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frames
                cv2.imwrite(f'color_frame_{frame_count}.png', color_image)
                cv2.imwrite(f'depth_frame_{frame_count}.png', depth_colormap)
                print(f"✅ Frames saved as color_frame_{frame_count}.png and depth_frame_{frame_count}.png")
            
            frame_count += 1
            
            # Print FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"📊 FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Test completed:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("✅ Camera test finished successfully")

if __name__ == "__main__":
    main()
