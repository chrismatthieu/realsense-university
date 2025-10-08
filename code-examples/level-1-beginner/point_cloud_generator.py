#!/usr/bin/env python3
"""
RealSense Point Cloud Generator
==============================

This script demonstrates how to generate and save 3D point clouds from RealSense cameras.
It includes point cloud filtering, visualization, and export functionality.

Features:
- Real-time point cloud generation
- Point cloud filtering and processing
- 3D visualization with Open3D
- Export to PLY format
- Interactive point cloud exploration

Author: RealSense University
Level: 1 (Beginner)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
from typing import Optional, Tuple

class PointCloudGenerator:
    """Generate and process point clouds from RealSense data"""
    
    def __init__(self):
        self.point_cloud = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        
    def generate_point_cloud(self, depth_frame, color_frame) -> Optional[o3d.geometry.PointCloud]:
        """
        Generate point cloud from depth and color frames
        
        Args:
            depth_frame: RealSense depth frame
            color_frame: RealSense color frame
            
        Returns:
            Open3D point cloud or None if failed
        """
        try:
            # Align depth to color
            aligned_frames = self.align.process(rs.composite_frame([depth_frame, color_frame]))
            aligned_depth = aligned_frames.get_depth_frame()
            aligned_color = aligned_frames.get_color_frame()
            
            # Generate point cloud
            self.point_cloud.map_to(aligned_color)
            points = self.point_cloud.calculate(aligned_depth)
            
            # Get point cloud data
            vertices = np.asanyarray(points.get_vertices())
            colors = np.asanyarray(points.get_colors())
            
            # Filter out invalid points
            valid_mask = vertices['f2'] > 0  # Filter by depth > 0
            valid_vertices = vertices[valid_mask]
            valid_colors = colors[valid_mask]
            
            if len(valid_vertices) == 0:
                return None
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(valid_vertices)
            pcd.colors = o3d.utility.Vector3dVector(valid_colors)
            
            return pcd
            
        except Exception as e:
            print(f"❌ Error generating point cloud: {e}")
            return None
    
    def filter_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Filter point cloud to remove noise and outliers
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Filtered point cloud
        """
        try:
            # Remove statistical outliers
            pcd_filtered, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            
            # Remove radius outliers
            pcd_filtered, _ = pcd_filtered.remove_radius_outlier(
                nb_points=16, radius=0.05
            )
            
            # Voxel downsampling
            pcd_downsampled = pcd_filtered.voxel_down_sample(0.01)
            
            print(f"✅ Point cloud filtered: {len(pcd.points)} -> {len(pcd_downsampled.points)} points")
            
            return pcd_downsampled
            
        except Exception as e:
            print(f"❌ Error filtering point cloud: {e}")
            return pcd
    
    def save_point_cloud(self, pcd: o3d.geometry.PointCloud, filename: str) -> bool:
        """
        Save point cloud to file
        
        Args:
            pcd: Point cloud to save
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = o3d.io.write_point_cloud(filename, pcd)
            if success:
                print(f"✅ Point cloud saved as {filename}")
                return True
            else:
                print(f"❌ Failed to save point cloud to {filename}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving point cloud: {e}")
            return False
    
    def visualize_point_cloud(self, pcd: o3d.geometry.PointCloud, window_name: str = "Point Cloud"):
        """
        Visualize point cloud with Open3D
        
        Args:
            pcd: Point cloud to visualize
            window_name: Window name
        """
        try:
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name)
            vis.add_geometry(pcd)
            
            # Set rendering options
            render_option = vis.get_render_option()
            render_option.point_size = 2.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            
            # Run visualizer
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            print(f"❌ Error visualizing point cloud: {e}")

class RealSenseCamera:
    """RealSense camera handler for point cloud generation"""
    
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
    
    def get_frames(self) -> Tuple[Optional[rs.depth_frame], Optional[rs.color_frame]]:
        """Get current frames from camera"""
        if not self.is_streaming:
            return None, None
            
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            return depth_frame, color_frame
                
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
    """Main function for point cloud generation demo"""
    print("☁️ RealSense Point Cloud Generator")
    print("=" * 35)
    
    # Initialize components
    camera = RealSenseCamera()
    pc_generator = PointCloudGenerator()
    
    if not camera.initialize():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    print("\n🎯 Point Cloud Generation Started!")
    print("Press 'g' to generate point cloud")
    print("Press 'v' to visualize last point cloud")
    print("Press 's' to save last point cloud")
    print("Press 'q' to quit")
    
    last_point_cloud = None
    frame_count = 0
    
    try:
        while True:
            # Get frames
            depth_frame, color_frame = camera.get_frames()
            
            if depth_frame is None or color_frame is None:
                print("❌ Failed to get frames")
                break
            
            # Convert frames to images for display
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Create depth colormap for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Display images
            cv2.imshow('Color Stream', color_image)
            cv2.imshow('Depth Stream', depth_colormap)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('g'):
                # Generate point cloud
                print("🔄 Generating point cloud...")
                point_cloud = pc_generator.generate_point_cloud(depth_frame, color_frame)
                
                if point_cloud is not None:
                    print(f"✅ Generated point cloud with {len(point_cloud.points)} points")
                    
                    # Filter point cloud
                    filtered_pc = pc_generator.filter_point_cloud(point_cloud)
                    last_point_cloud = filtered_pc
                    
                    print(f"✅ Filtered point cloud has {len(filtered_pc.points)} points")
                else:
                    print("❌ Failed to generate point cloud")
                    
            elif key == ord('v') and last_point_cloud is not None:
                # Visualize point cloud
                print("🎨 Opening point cloud visualizer...")
                pc_generator.visualize_point_cloud(last_point_cloud)
                
            elif key == ord('s') and last_point_cloud is not None:
                # Save point cloud
                timestamp = int(time.time())
                filename = f"point_cloud_{timestamp}.ply"
                pc_generator.save_point_cloud(last_point_cloud, filename)
                
            frame_count += 1
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"📊 Frames processed: {frame_count}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Point cloud generation interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        print("✅ Point cloud generation finished successfully")

if __name__ == "__main__":
    main()
