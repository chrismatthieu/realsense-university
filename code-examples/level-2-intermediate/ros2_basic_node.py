#!/usr/bin/env python3
"""
RealSense ROS2 Basic Node
========================

This script demonstrates how to create a basic ROS2 node that publishes
RealSense camera data. It shows how to integrate RealSense cameras with ROS2
and publish RGB, depth, and point cloud data.

Features:
- ROS2 node creation and management
- RealSense camera integration
- Topic publishing (RGB, depth, point clouds)
- Parameter configuration
- Error handling and logging

Author: RealSense University
Level: 2 (Intermediate)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs
from typing import Optional

class RealSenseROS2Node(Node):
    """RealSense ROS2 node for publishing camera data"""
    
    def __init__(self):
        super().__init__('realsense_camera_node')
        
        # Declare parameters
        self.declare_parameter('camera_name', 'camera')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('enable_color', True)
        self.declare_parameter('enable_depth', True)
        self.declare_parameter('enable_pointcloud', True)
        
        # Get parameters
        self.camera_name = self.get_parameter('camera_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.enable_color = self.get_parameter('enable_color').value
        self.enable_depth = self.get_parameter('enable_depth').value
        self.enable_pointcloud = self.get_parameter('enable_pointcloud').value
        
        # Initialize components
        self.bridge = CvBridge()
        self.pipeline = None
        self.config = None
        self.point_cloud = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        
        # Create publishers
        self.create_publishers()
        
        # Initialize camera
        if not self.initialize_camera():
            self.get_logger().error("Failed to initialize RealSense camera")
            return
        
        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.fps, self.publish_data)
        
        self.get_logger().info(f"RealSense ROS2 node started with camera: {self.camera_name}")
    
    def create_publishers(self):
        """Create ROS2 publishers for different data types"""
        # Color image publisher
        if self.enable_color:
            self.color_pub = self.create_publisher(
                Image, f'/{self.camera_name}/color/image_raw', 10
            )
            self.color_info_pub = self.create_publisher(
                CameraInfo, f'/{self.camera_name}/color/camera_info', 10
            )
        
        # Depth image publisher
        if self.enable_depth:
            self.depth_pub = self.create_publisher(
                Image, f'/{self.camera_name}/depth/image_rect_raw', 10
            )
            self.depth_info_pub = self.create_publisher(
                CameraInfo, f'/{self.camera_name}/depth/camera_info', 10
            )
        
        # Point cloud publisher
        if self.enable_pointcloud:
            self.pointcloud_pub = self.create_publisher(
                PointCloud2, f'/{self.camera_name}/depth/color/points', 10
            )
    
    def initialize_camera(self) -> bool:
        """Initialize RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            if self.enable_color:
                self.config.enable_stream(
                    rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
                )
            
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
                )
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            if self.enable_color:
                color_profile = profile.get_stream(rs.stream.color)
                self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            if self.enable_depth:
                depth_profile = profile.get_stream(rs.stream.depth)
                self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
            self.get_logger().info("RealSense camera initialized successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize camera: {e}")
            return False
    
    def publish_data(self):
        """Main publishing function called by timer"""
        try:
            # Get frames
            frames = self.pipeline.wait_for_frames()
            
            # Align frames
            aligned_frames = self.align.process(frames)
            
            # Get individual frames
            color_frame = aligned_frames.get_color_frame() if self.enable_color else None
            depth_frame = aligned_frames.get_depth_frame() if self.enable_depth else None
            
            # Publish color data
            if color_frame and self.enable_color:
                self.publish_color_data(color_frame)
            
            # Publish depth data
            if depth_frame and self.enable_depth:
                self.publish_depth_data(depth_frame)
            
            # Publish point cloud
            if color_frame and depth_frame and self.enable_pointcloud:
                self.publish_pointcloud_data(color_frame, depth_frame)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing data: {e}")
    
    def publish_color_data(self, color_frame):
        """Publish color image and camera info"""
        try:
            # Convert to OpenCV image
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert to ROS2 Image message
            color_msg = self.bridge.cv2_to_imgmsg(color_image, 'bgr8')
            color_msg.header.frame_id = self.frame_id
            color_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Publish color image
            self.color_pub.publish(color_msg)
            
            # Create and publish camera info
            color_info = self.create_camera_info(self.color_intrinsics, color_msg.header)
            self.color_info_pub.publish(color_info)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing color data: {e}")
    
    def publish_depth_data(self, depth_frame):
        """Publish depth image and camera info"""
        try:
            # Convert to OpenCV image
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert to ROS2 Image message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, '16UC1')
            depth_msg.header.frame_id = self.frame_id
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Publish depth image
            self.depth_pub.publish(depth_msg)
            
            # Create and publish camera info
            depth_info = self.create_camera_info(self.depth_intrinsics, depth_msg.header)
            self.depth_info_pub.publish(depth_info)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing depth data: {e}")
    
    def publish_pointcloud_data(self, color_frame, depth_frame):
        """Publish point cloud data"""
        try:
            # Generate point cloud
            self.point_cloud.map_to(color_frame)
            points = self.point_cloud.calculate(depth_frame)
            
            # Get point cloud data
            vertices = np.asanyarray(points.get_vertices())
            colors = np.asanyarray(points.get_colors())
            
            # Filter valid points
            valid_mask = vertices['f2'] > 0
            valid_vertices = vertices[valid_mask]
            valid_colors = colors[valid_mask]
            
            if len(valid_vertices) > 0:
                # Create PointCloud2 message
                pointcloud_msg = self.create_pointcloud2_msg(
                    valid_vertices, valid_colors, self.frame_id
                )
                
                # Publish point cloud
                self.pointcloud_pub.publish(pointcloud_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error publishing point cloud: {e}")
    
    def create_camera_info(self, intrinsics, header: Header) -> CameraInfo:
        """Create CameraInfo message from intrinsics"""
        camera_info = CameraInfo()
        camera_info.header = header
        
        # Set camera parameters
        camera_info.width = intrinsics.width
        camera_info.height = intrinsics.height
        camera_info.distortion_model = 'plumb_bob'
        
        # Set camera matrix
        camera_info.k = [
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        # Set distortion coefficients
        camera_info.d = list(intrinsics.coeffs)
        
        # Set projection matrix
        camera_info.p = [
            intrinsics.fx, 0.0, intrinsics.ppx, 0.0,
            0.0, intrinsics.fy, intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        return camera_info
    
    def create_pointcloud2_msg(self, vertices, colors, frame_id: str) -> PointCloud2:
        """Create PointCloud2 message from vertices and colors"""
        from sensor_msgs.msg import PointField
        
        # Create header
        header = Header()
        header.frame_id = frame_id
        header.stamp = self.get_clock().now().to_msg()
        
        # Create point cloud data
        points_data = []
        for i in range(len(vertices)):
            # Add position (x, y, z)
            points_data.extend([vertices[i][0], vertices[i][1], vertices[i][2]])
            # Add color (r, g, b)
            points_data.extend([colors[i][0], colors[i][1], colors[i][2]])
        
        # Create PointCloud2 message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(vertices)
        pointcloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
        ]
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 15  # 3 * 4 + 3 * 1
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.data = np.array(points_data, dtype=np.uint8).tobytes()
        pointcloud_msg.is_dense = True
        
        return pointcloud_msg
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.pipeline:
            self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    
    try:
        node = RealSenseROS2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
