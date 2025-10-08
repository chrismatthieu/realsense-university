#!/usr/bin/env python3
"""
RealSense Distance Measurement Tool
===================================

This script implements a real-time distance measurement tool using RealSense cameras.
It allows users to measure distances to objects by clicking on the depth image.

Features:
- Real-time distance measurement
- Visual crosshair for precise targeting
- Distance display in millimeters and inches
- Save measurements to CSV file
- Simple keyboard and mouse controls

Author: RealSense University
Level: 1 (Beginner)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import datetime
import os
from typing import Tuple, Optional

class DistanceMeasurer:
    """Distance measurement and data handling"""
    
    def __init__(self, data_file='measurements.csv'):
        """
        Initialize distance measurer
        
        Args:
            data_file (str): CSV file to save measurements
        """
        self.data_file = data_file
        self.measurements = []
        
    def measure_distance(self, depth_image: np.ndarray, x: int, y: int) -> Optional[float]:
        """
        Measure distance at specific pixel coordinates
        
        Args:
            depth_image (np.ndarray): Depth image
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Distance in millimeters or None if invalid
        """
        try:
            # Check bounds
            if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
                return None
            
            # Get depth value
            depth_value = depth_image[y, x]
            
            # Check if depth is valid (not zero)
            if depth_value == 0:
                return None
            
            # Convert to millimeters (assuming depth units are in mm)
            distance_mm = depth_value
            return distance_mm
            
        except Exception as e:
            print(f"❌ Error measuring distance: {e}")
            return None
    
    def mm_to_inches(self, mm: float) -> float:
        """
        Convert millimeters to inches
        
        Args:
            mm (float): Distance in millimeters
            
        Returns:
            float: Distance in inches
        """
        return mm / 25.4
    
    def get_measurement_info(self, depth_image: np.ndarray, x: int, y: int) -> dict:
        """
        Get comprehensive measurement information
        
        Args:
            depth_image (np.ndarray): Depth image
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            dict: Measurement information
        """
        distance_mm = self.measure_distance(depth_image, x, y)
        
        if distance_mm is None:
            return {
                'distance_mm': None,
                'distance_inches': None,
                'valid': False,
                'message': 'No valid depth data'
            }
        
        distance_inches = self.mm_to_inches(distance_mm)
        
        return {
            'distance_mm': distance_mm,
            'distance_inches': distance_inches,
            'valid': True,
            'message': f'{distance_mm:.1f}mm ({distance_inches:.2f}")'
        }
    
    def save_measurement(self, x: int, y: int, distance_mm: float, timestamp: str = None):
        """
        Save measurement to CSV file
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            distance_mm (float): Distance in millimeters
            timestamp (str): Timestamp (optional)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        measurement = {
            'timestamp': timestamp,
            'x': x,
            'y': y,
            'distance_mm': distance_mm,
            'distance_inches': self.mm_to_inches(distance_mm)
        }
        
        self.measurements.append(measurement)
        
        # Save to CSV
        try:
            with open(self.data_file, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'x', 'y', 'distance_mm', 'distance_inches']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is empty
                if csvfile.tell() == 0:
                    writer.writeheader()
                
                writer.writerow(measurement)
                
            print(f"✅ Measurement saved: {measurement['distance_mm']:.1f}mm")
            
        except Exception as e:
            print(f"❌ Error saving measurement: {e}")

class MeasurementUI:
    """User interface for distance measurement"""
    
    def __init__(self, window_name="RealSense Distance Measurement"):
        """
        Initialize measurement UI
        
        Args:
            window_name (str): Window name
        """
        self.window_name = window_name
        self.crosshair_size = 20
        self.crosshair_thickness = 2
        self.crosshair_color = (0, 255, 0)  # Green
        self.text_color = (255, 255, 255)   # White
        self.text_bg_color = (0, 0, 0)      # Black
        
    def draw_crosshair(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Draw crosshair at specified coordinates
        
        Args:
            image (np.ndarray): Image to draw on
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            np.ndarray: Image with crosshair
        """
        # Draw horizontal line
        cv2.line(image, 
                (x - self.crosshair_size, y), 
                (x + self.crosshair_size, y), 
                self.crosshair_color, 
                self.crosshair_thickness)
        
        # Draw vertical line
        cv2.line(image, 
                (x, y - self.crosshair_size), 
                (x, y + self.crosshair_size), 
                self.crosshair_color, 
                self.crosshair_thickness)
        
        return image
    
    def draw_text_with_background(self, image: np.ndarray, text: str, position: Tuple[int, int]) -> np.ndarray:
        """
        Draw text with background for better visibility
        
        Args:
            image (np.ndarray): Image to draw on
            text (str): Text to draw
            position (tuple): (x, y) position
            
        Returns:
            np.ndarray: Image with text
        """
        x, y = position
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(image, 
                     (x, y - text_height - baseline), 
                     (x + text_width, y + baseline), 
                     self.text_bg_color, 
                     -1)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, font_scale, self.text_color, thickness)
        
        return image
    
    def draw_measurement_info(self, image: np.ndarray, measurement_info: dict, x: int, y: int) -> np.ndarray:
        """
        Draw measurement information on image
        
        Args:
            image (np.ndarray): Image to draw on
            measurement_info (dict): Measurement information
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            np.ndarray: Image with measurement info
        """
        if not measurement_info['valid']:
            text = measurement_info['message']
            self.draw_text_with_background(image, text, (10, 30))
            return image
        
        # Draw distance information
        distance_text = f"Distance: {measurement_info['message']}"
        self.draw_text_with_background(image, distance_text, (10, 30))
        
        # Draw coordinates
        coord_text = f"Position: ({x}, {y})"
        self.draw_text_with_background(image, coord_text, (10, 60))
        
        # Draw instructions
        instructions = [
            "Controls:",
            "Mouse: Move crosshair",
            "Click: Measure distance",
            "S: Save measurement",
            "Q: Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            self.draw_text_with_background(image, instruction, (10, 90 + i * 25))
        
        return image
    
    def create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Create colored depth map for visualization
        
        Args:
            depth_image (np.ndarray): Raw depth image
            
        Returns:
            np.ndarray: Colored depth map
        """
        # Normalize depth image
        depth_normalized = cv2.convertScaleAbs(depth_image, alpha=0.03)
        
        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colormap

def main():
    """Main application function"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize components
    camera = RealSenseCamera()
    measurer = DistanceMeasurer('data/measurements.csv')
    ui = MeasurementUI()
    
    # Initialize camera
    if not camera.initialize():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    # Get camera info
    camera_info = camera.get_camera_info()
    if camera_info:
        print(f"📷 Camera: {camera_info['name']}")
        print(f"🔢 Serial: {camera_info['serial']}")
        print(f"⚙️ Firmware: {camera_info['firmware']}")
    
    # Mouse callback for crosshair movement
    crosshair_x, crosshair_y = 320, 240  # Center of 640x480 image
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal crosshair_x, crosshair_y
        if event == cv2.EVENT_MOUSEMOVE:
            crosshair_x, crosshair_y = x, y
    
    # Set mouse callback
    cv2.setMouseCallback('Color Stream', mouse_callback)
    cv2.setMouseCallback('Depth Stream', mouse_callback)
    
    print("\n🎯 Distance Measurement Tool Started!")
    print("Move your mouse to position the crosshair, click to measure, press 's' to save, 'q' to quit")
    
    try:
        while True:
            # Get frames from camera
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                print("❌ Failed to get frames from camera")
                break
            
            # Get measurement at crosshair position
            measurement_info = measurer.get_measurement_info(
                depth_image, crosshair_x, crosshair_y
            )
            
            # Create depth colormap
            depth_colormap = ui.create_depth_colormap(depth_image)
            
            # Draw crosshair on both images
            color_with_crosshair = ui.draw_crosshair(color_image.copy(), crosshair_x, crosshair_y)
            depth_with_crosshair = ui.draw_crosshair(depth_colormap.copy(), crosshair_x, crosshair_y)
            
            # Draw measurement info
            color_with_info = ui.draw_measurement_info(color_with_crosshair, measurement_info, crosshair_x, crosshair_y)
            depth_with_info = ui.draw_measurement_info(depth_with_crosshair, measurement_info, crosshair_x, crosshair_y)
            
            # Display images
            cv2.imshow('Color Stream', color_with_info)
            cv2.imshow('Depth Stream', depth_with_info)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and measurement_info['valid']:
                # Save measurement
                measurer.save_measurement(
                    crosshair_x, crosshair_y, measurement_info['distance_mm']
                )
            
            elif key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Application interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("✅ Application closed successfully")

if __name__ == "__main__":
    main()
