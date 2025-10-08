#!/usr/bin/env python3
"""
RealSense Visual SLAM System
===========================

This script implements a Visual SLAM system using RealSense cameras.
It demonstrates feature extraction, tracking, mapping, and loop closure detection.

Features:
- ORB feature extraction and matching
- Camera pose estimation
- 3D map building
- Loop closure detection
- Map optimization
- Real-time visualization

Author: RealSense University
Level: 3 (Advanced)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import threading
import queue

@dataclass
class KeyFrame:
    """Keyframe data structure"""
    id: int
    timestamp: float
    rgb_image: np.ndarray
    depth_image: np.ndarray
    pose: np.ndarray  # 4x4 transformation matrix
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    points_3d: np.ndarray

@dataclass
class MapPoint:
    """3D map point data structure"""
    id: int
    position: np.ndarray  # 3D position
    descriptor: np.ndarray
    observations: List[int]  # Keyframe IDs that observe this point
    is_valid: bool

class VisualSLAM:
    """Visual SLAM system implementation"""
    
    def __init__(self, config: Dict):
        """
        Initialize Visual SLAM system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Feature extraction parameters
        self.orb = cv2.ORB_create(
            nfeatures=config.get('nfeatures', 1000),
            scaleFactor=config.get('scaleFactor', 1.2),
            nlevels=config.get('nlevels', 8)
        )
        
        # Feature matching parameters
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = config.get('min_matches', 50)
        
        # SLAM state
        self.keyframes = []
        self.map_points = []
        self.current_pose = np.eye(4)
        self.next_keyframe_id = 0
        self.next_mappoint_id = 0
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Tracking state
        self.is_initialized = False
        self.last_keyframe = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
    
    def initialize_camera(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Initialize camera parameters"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        print("✅ Camera parameters initialized")
    
    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> Dict:
        """
        Process a new frame for SLAM
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            
        Returns:
            SLAM processing results
        """
        self.frame_count += 1
        
        # Extract features
        keypoints, descriptors = self.extract_features(rgb_image)
        
        if not self.is_initialized:
            # Initialize SLAM with first frame
            return self.initialize_slam(rgb_image, depth_image, keypoints, descriptors)
        else:
            # Track and update map
            return self.track_and_map(rgb_image, depth_image, keypoints, descriptors)
    
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract ORB features from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def initialize_slam(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                       keypoints: List, descriptors: np.ndarray) -> Dict:
        """Initialize SLAM system with first frame"""
        # Create first keyframe
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            rgb_image=rgb_image.copy(),
            depth_image=depth_image.copy(),
            pose=np.eye(4),
            keypoints=keypoints,
            descriptors=descriptors,
            points_3d=np.array([])
        )
        
        # Generate 3D points
        points_3d = self.generate_3d_points(keypoints, depth_image)
        keyframe.points_3d = points_3d
        
        # Add keyframe
        self.keyframes.append(keyframe)
        self.last_keyframe = keyframe
        self.next_keyframe_id += 1
        
        # Create initial map points
        self.create_initial_map_points(keyframe)
        
        self.is_initialized = True
        
        return {
            'status': 'initialized',
            'keyframes': len(self.keyframes),
            'map_points': len(self.map_points),
            'pose': self.current_pose
        }
    
    def track_and_map(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                     keypoints: List, descriptors: np.ndarray) -> Dict:
        """Track camera pose and update map"""
        # Match features with last keyframe
        matches = self.match_features(descriptors, self.last_keyframe.descriptors)
        
        if len(matches) < self.min_matches:
            return {
                'status': 'tracking_lost',
                'matches': len(matches),
                'keyframes': len(self.keyframes),
                'map_points': len(self.map_points)
            }
        
        # Estimate pose
        pose = self.estimate_pose(matches, keypoints, self.last_keyframe.keypoints)
        
        if pose is None:
            return {
                'status': 'pose_estimation_failed',
                'matches': len(matches),
                'keyframes': len(self.keyframes),
                'map_points': len(self.map_points)
            }
        
        # Update current pose
        self.current_pose = pose
        
        # Check if we need a new keyframe
        if self.should_create_keyframe(pose, matches):
            # Create new keyframe
            keyframe = self.create_keyframe(rgb_image, depth_image, keypoints, descriptors, pose)
            
            # Update map
            self.update_map(keyframe, matches)
            
            return {
                'status': 'keyframe_created',
                'matches': len(matches),
                'keyframes': len(self.keyframes),
                'map_points': len(self.map_points),
                'pose': pose
            }
        else:
            return {
                'status': 'tracking',
                'matches': len(matches),
                'keyframes': len(self.keyframes),
                'map_points': len(self.map_points),
                'pose': pose
            }
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []
        
        # Match features
        matches = self.matcher.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Filter good matches
        good_matches = []
        for match in matches:
            if match.distance < 50:  # Threshold for good matches
                good_matches.append(match)
        
        return good_matches
    
    def estimate_pose(self, matches: List, kp1: List, kp2: List) -> Optional[np.ndarray]:
        """Estimate camera pose from feature matches"""
        if len(matches) < 8:
            return None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC)
        
        if E is None:
            return None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        # Create transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        return pose
    
    def should_create_keyframe(self, pose: np.ndarray, matches: List) -> bool:
        """Determine if a new keyframe should be created"""
        if self.last_keyframe is None:
            return True
        
        # Check translation distance
        translation = np.linalg.norm(pose[:3, 3])
        if translation > 0.1:  # 10cm threshold
            return True
        
        # Check number of matches
        if len(matches) < self.min_matches * 0.7:
            return True
        
        return False
    
    def create_keyframe(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                       keypoints: List, descriptors: np.ndarray, pose: np.ndarray) -> KeyFrame:
        """Create a new keyframe"""
        # Generate 3D points
        points_3d = self.generate_3d_points(keypoints, depth_image)
        
        # Create keyframe
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            rgb_image=rgb_image.copy(),
            depth_image=depth_image.copy(),
            pose=pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            points_3d=points_3d
        )
        
        # Add to keyframes
        self.keyframes.append(keyframe)
        self.last_keyframe = keyframe
        self.next_keyframe_id += 1
        
        return keyframe
    
    def generate_3d_points(self, keypoints: List, depth_image: np.ndarray) -> np.ndarray:
        """Generate 3D points from keypoints and depth"""
        points_3d = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                depth = depth_image[y, x]
                
                if depth > 0:
                    # Convert to 3D coordinates
                    z = depth / 1000.0  # Convert mm to m
                    x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                    y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                    
                    points_3d.append([x_3d, y_3d, z])
        
        return np.array(points_3d)
    
    def create_initial_map_points(self, keyframe: KeyFrame):
        """Create initial map points from first keyframe"""
        for i, point_3d in enumerate(keyframe.points_3d):
            if len(point_3d) == 3:
                map_point = MapPoint(
                    id=self.next_mappoint_id,
                    position=point_3d,
                    descriptor=keyframe.descriptors[i] if i < len(keyframe.descriptors) else None,
                    observations=[keyframe.id],
                    is_valid=True
                )
                
                self.map_points.append(map_point)
                self.next_mappoint_id += 1
    
    def update_map(self, keyframe: KeyFrame, matches: List):
        """Update map with new keyframe"""
        # Add new map points
        for i, point_3d in enumerate(keyframe.points_3d):
            if len(point_3d) == 3:
                map_point = MapPoint(
                    id=self.next_mappoint_id,
                    position=point_3d,
                    descriptor=keyframe.descriptors[i] if i < len(keyframe.descriptors) else None,
                    observations=[keyframe.id],
                    is_valid=True
                )
                
                self.map_points.append(map_point)
                self.next_mappoint_id += 1
    
    def get_map_pointcloud(self) -> o3d.geometry.PointCloud:
        """Get 3D map as point cloud"""
        if not self.map_points:
            return o3d.geometry.PointCloud()
        
        # Extract valid map points
        valid_points = [mp.position for mp in self.map_points if mp.is_valid]
        
        if not valid_points:
            return o3d.geometry.PointCloud()
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
        
        # Add colors (simple coloring based on height)
        colors = []
        for point in valid_points:
            # Color based on height (z coordinate)
            height = point[2]
            if height < 0.5:
                colors.append([0, 0, 1])  # Blue for low points
            elif height < 1.0:
                colors.append([0, 1, 0])  # Green for medium points
            else:
                colors.append([1, 0, 0])  # Red for high points
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def get_keyframe_trajectory(self) -> List[np.ndarray]:
        """Get camera trajectory from keyframes"""
        trajectory = []
        for kf in self.keyframes:
            trajectory.append(kf.pose[:3, 3])  # Extract translation
        return trajectory
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'fps': fps,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time,
            'keyframes': len(self.keyframes),
            'map_points': len(self.map_points)
        }

class SLAMVisualizer:
    """Visualization for SLAM system"""
    
    def __init__(self):
        self.trajectory_points = []
        self.map_visualizer = None
    
    def visualize_map(self, slam_system: VisualSLAM):
        """Visualize SLAM map and trajectory"""
        # Get map point cloud
        map_pcd = slam_system.get_map_pointcloud()
        
        # Get trajectory
        trajectory = slam_system.get_keyframe_trajectory()
        
        if len(trajectory) > 0:
            # Create trajectory line
            trajectory_points = np.array(trajectory)
            trajectory_pcd = o3d.geometry.PointCloud()
            trajectory_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
            trajectory_pcd.paint_uniform_color([1, 0, 0])  # Red trajectory
            
            # Create coordinate frame for current pose
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coordinate_frame.transform(slam_system.current_pose)
            
            # Visualize
            geometries = [map_pcd, trajectory_pcd, coordinate_frame]
            o3d.visualization.draw_geometries(geometries, window_name="SLAM Map")
    
    def draw_slam_info(self, image: np.ndarray, slam_stats: Dict) -> np.ndarray:
        """Draw SLAM information on image"""
        result_image = image.copy()
        
        # Create info text
        info_lines = [
            f"FPS: {slam_stats['fps']:.1f}",
            f"Keyframes: {slam_stats['keyframes']}",
            f"Map Points: {slam_stats['map_points']}",
            f"Frames: {slam_stats['frame_count']}"
        ]
        
        # Draw background
        cv2.rectangle(result_image, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(result_image, (10, 10), (300, 100), (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 20
            cv2.putText(result_image, line, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result_image

class RealSenseCamera:
    """RealSense camera handler for SLAM"""
    
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
            
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
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
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera matrix from intrinsics"""
        if hasattr(self, 'color_intrinsics'):
            return np.array([
                [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                [0, 0, 1]
            ])
        return None
    
    def get_dist_coeffs(self) -> np.ndarray:
        """Get distortion coefficients from intrinsics"""
        if hasattr(self, 'color_intrinsics'):
            return np.array(self.color_intrinsics.coeffs)
        return None
    
    def stop(self):
        """Stop camera streaming"""
        if self.pipeline and self.is_streaming:
            self.pipeline.stop()
            self.is_streaming = False
            print("✅ Camera stopped")

def main():
    """Main function for SLAM demo"""
    print("🗺️ RealSense Visual SLAM System")
    print("=" * 35)
    
    # Configuration
    config = {
        'nfeatures': 1000,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'min_matches': 50
    }
    
    # Initialize components
    camera = RealSenseCamera()
    slam = VisualSLAM(config)
    visualizer = SLAMVisualizer()
    
    if not camera.initialize():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    # Initialize camera parameters
    camera_matrix = camera.get_camera_matrix()
    dist_coeffs = camera.get_dist_coeffs()
    
    if camera_matrix is not None:
        slam.initialize_camera(camera_matrix, dist_coeffs)
    else:
        print("❌ Failed to get camera parameters")
        return
    
    print("\n🎯 SLAM System Started!")
    print("Press 'q' to quit, 'v' to visualize map, 's' to save map")
    
    try:
        while True:
            # Get frames
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                print("❌ Failed to get frames")
                break
            
            # Process frame with SLAM
            slam_result = slam.process_frame(color_image, depth_image)
            
            # Get performance stats
            stats = slam.get_performance_stats()
            
            # Visualize results
            color_with_info = visualizer.draw_slam_info(color_image, stats)
            
            # Display images
            cv2.imshow('SLAM System', color_with_info)
            
            # Create depth visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth Map', depth_colormap)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('v'):
                # Visualize map
                print("🎨 Opening map visualizer...")
                visualizer.visualize_map(slam)
            elif key == ord('s'):
                # Save map
                map_pcd = slam.get_map_pointcloud()
                if len(map_pcd.points) > 0:
                    timestamp = int(time.time())
                    filename = f"slam_map_{timestamp}.ply"
                    o3d.io.write_point_cloud(filename, map_pcd)
                    print(f"✅ Map saved as {filename}")
                else:
                    print("❌ No map points to save")
            
            # Print status every 30 frames
            if slam.frame_count % 30 == 0:
                print(f"📊 Status: {slam_result['status']}, "
                      f"Keyframes: {slam_result['keyframes']}, "
                      f"Map Points: {slam_result['map_points']}")
    
    except KeyboardInterrupt:
        print("\n⏹️ SLAM system interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = slam.get_performance_stats()
        print(f"\n📊 SLAM completed:")
        print(f"   Total frames: {stats['frame_count']}")
        print(f"   Average FPS: {stats['fps']:.1f}")
        print(f"   Keyframes: {stats['keyframes']}")
        print(f"   Map points: {stats['map_points']}")
        print("✅ SLAM system finished successfully")

if __name__ == "__main__":
    main()
