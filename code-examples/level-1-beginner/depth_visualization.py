#!/usr/bin/env python3
"""
RealSense Depth Visualization Tool
==================================

This script demonstrates various ways to visualize depth data from RealSense cameras.
It shows different colormaps, depth range adjustments, and interactive controls.

Features:
- Multiple depth visualization modes
- Interactive depth range adjustment
- Real-time depth statistics
- Save depth visualizations
- Keyboard controls for different modes

Author: RealSense University
Level: 1 (Beginner)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from typing import Dict, Tuple

class DepthVisualizer:
    """Advanced depth visualization with multiple modes"""
    
    def __init__(self):
        self.colormap_modes = {
            'JET': cv2.COLORMAP_JET,
            'HOT': cv2.COLORMAP_HOT,
            'COOL': cv2.COLORMAP_COOL,
            'RAINBOW': cv2.COLORMAP_RAINBOW,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS,
            'PLASMA': cv2.COLORMAP_PLASMA,
            'INFERNO': cv2.COLORMAP_INFERNO,
            'MAGMA': cv2.COLORMAP_MAGMA
        }
        self.current_mode = 'JET'
        self.depth_scale = 0.03  # Alpha value for depth scaling
        self.min_depth = 0
        self.max_depth = 5000  # 5 meters in mm
        
    def visualize_depth(self, depth_image: np.ndarray, mode: str = None) -> np.ndarray:
        """
        Visualize depth image with specified colormap
        
        Args:
            depth_image (np.ndarray): Raw depth image
            mode (str): Colormap mode
            
        Returns:
            np.ndarray: Colored depth image
        """
        if mode is None:
            mode = self.current_mode
            
        # Normalize depth image
        depth_normalized = cv2.convertScaleAbs(depth_image, alpha=self.depth_scale)
        
        # Apply colormap
        if mode in self.colormap_modes:
            depth_colormap = cv2.applyColorMap(depth_normalized, self.colormap_modes[mode])
        else:
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
        return depth_colormap
    
    def get_depth_statistics(self, depth_image: np.ndarray) -> Dict:
        """
        Calculate depth statistics
        
        Args:
            depth_image (np.ndarray): Raw depth image
            
        Returns:
            dict: Depth statistics
        """
        # Filter out invalid depths (0 values)
        valid_depths = depth_image[depth_image > 0]
        
        if len(valid_depths) == 0:
            return {
                'min': 0, 'max': 0, 'mean': 0, 'std': 0,
                'valid_pixels': 0, 'total_pixels': depth_image.size
            }
        
        return {
            'min': np.min(valid_depths),
            'max': np.max(valid_depths),
            'mean': np.mean(valid_depths),
            'std': np.std(valid_depths),
            'valid_pixels': len(valid_depths),
            'total_pixels': depth_image.size
        }
    
    def draw_depth_info(self, image: np.ndarray, stats: Dict, mode: str) -> np.ndarray:
        """
        Draw depth information on image
        
        Args:
            image (np.ndarray): Image to draw on
            stats (dict): Depth statistics
            mode (str): Current visualization mode
            
        Returns:
            np.ndarray: Image with depth info
        """
        # Create info text
        info_lines = [
            f"Mode: {mode}",
            f"Min Depth: {stats['min']:.0f}mm",
            f"Max Depth: {stats['max']:.0f}mm",
            f"Mean Depth: {stats['mean']:.0f}mm",
            f"Valid Pixels: {stats['valid_pixels']}/{stats['total_pixels']}",
            "",
            "Controls:",
            "1-8: Change colormap",
            "Q: Quit",
            "S: Save image"
        ]
        
        # Draw background rectangle
        cv2.rectangle(image, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 20
            cv2.putText(image, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def adjust_depth_range(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Adjust depth range for better visualization
        
        Args:
            depth_image (np.ndarray): Raw depth image
            
        Returns:
            np.ndarray: Adjusted depth image
        """
        # Clip depth values to range
        depth_clipped = np.clip(depth_image, self.min_depth, self.max_depth)
        
        # Normalize to 0-255 range
        if self.max_depth > self.min_depth:
            depth_normalized = ((depth_clipped - self.min_depth) / 
                              (self.max_depth - self.min_depth) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
        
        return depth_normalized

class RealSenseCamera:
    """RealSense camera handler for depth visualization"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.is_streaming = False
        
    def initialize(self):
        """Initialize the camera"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            
            self.pipeline.start(self.config)
            self.is_streaming = True
            
            print("✅ Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def get_frames(self):
        """Get current frames from camera"""
        if not self.is_streaming:
            return None, None
            
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                return depth_image, color_image
            else:
                return None, None
                
        except Exception as e:
            print(f"❌ Error getting frames: {e}")
            return None, None
    
    def stop(self):
        """Stop camera streaming"""
        if self.pipeline and self.is_streaming:
            self.pipeline.stop()
            self.is_streaming = False
            print("✅ Camera stopped")

def main():
    """Main function for depth visualization demo"""
    print("🎨 RealSense Depth Visualization Tool")
    print("=" * 40)
    
    # Initialize components
    camera = RealSenseCamera()
    visualizer = DepthVisualizer()
    
    if not camera.initialize():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    print("\n🎯 Depth Visualization Started!")
    print("Use number keys 1-8 to change colormap modes")
    print("Press 'q' to quit, 's' to save current visualization")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frames
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                print("❌ Failed to get frames")
                break
            
            # Visualize depth
            depth_colormap = visualizer.visualize_depth(depth_image)
            
            # Get depth statistics
            stats = visualizer.get_depth_statistics(depth_image)
            
            # Draw depth information
            depth_with_info = visualizer.draw_depth_info(
                depth_colormap.copy(), stats, visualizer.current_mode
            )
            
            # Display images
            cv2.imshow('Color Stream', color_image)
            cv2.imshow('Depth Visualization', depth_with_info)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current visualization
                timestamp = int(time.time())
                cv2.imwrite(f'depth_visualization_{timestamp}.png', depth_with_info)
                print(f"✅ Depth visualization saved as depth_visualization_{timestamp}.png")
            elif key >= ord('1') and key <= ord('8'):
                # Change colormap mode
                mode_index = key - ord('1')
                modes = list(visualizer.colormap_modes.keys())
                if mode_index < len(modes):
                    visualizer.current_mode = modes[mode_index]
                    print(f"🎨 Switched to {visualizer.current_mode} colormap")
            
            frame_count += 1
            
            # Print FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"📊 FPS: {fps:.1f}, Mode: {visualizer.current_mode}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Visualization interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Visualization completed:")
        print(f"   Total frames: {frame_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("✅ Depth visualization finished successfully")

if __name__ == "__main__":
    main()
