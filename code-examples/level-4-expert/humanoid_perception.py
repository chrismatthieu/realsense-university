#!/usr/bin/env python3
"""
RealSense Humanoid Robot Perception System
=========================================

This script implements an advanced perception system for humanoid robots using RealSense cameras.
It demonstrates multi-sensor fusion, balance control, manipulation planning, and human-robot interaction.

Features:
- Multi-camera sensor fusion
- Real-time balance control
- Advanced manipulation planning
- Human detection and interaction
- Safety monitoring and alerts
- Performance optimization

Author: RealSense University
Level: 4 (Expert)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import threading
import queue
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import mediapipe as mp
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import json

class SensorType(Enum):
    """Sensor types for humanoid robot"""
    HEAD_CAMERA = "head_camera"
    CHEST_CAMERA = "chest_camera"
    LEFT_HAND_CAMERA = "left_hand_camera"
    RIGHT_HAND_CAMERA = "right_hand_camera"
    IMU = "imu"
    JOINT_ENCODERS = "joint_encoders"

class BalanceState(Enum):
    """Balance states for humanoid robot"""
    STABLE = "stable"
    WARNING = "warning"
    CRITICAL = "critical"
    FALLING = "falling"

@dataclass
class SensorData:
    """Sensor data structure"""
    sensor_type: SensorType
    timestamp: float
    data: Dict
    confidence: float

@dataclass
class BalanceInfo:
    """Balance information structure"""
    state: BalanceState
    center_of_mass: np.ndarray
    support_polygon: np.ndarray
    stability_margin: float
    corrections: np.ndarray

@dataclass
class ManipulationTarget:
    """Manipulation target structure"""
    id: int
    position: np.ndarray
    orientation: np.ndarray
    grasp_points: List[np.ndarray]
    confidence: float
    object_type: str

class HumanoidPerceptionSystem:
    """Advanced perception system for humanoid robots"""
    
    def __init__(self, config: Dict):
        """
        Initialize humanoid perception system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Sensor management
        self.sensors = {}
        self.sensor_data_queue = queue.Queue()
        self.sensor_threads = {}
        
        # Perception modules
        self.balance_controller = BalanceController(config)
        self.manipulation_planner = ManipulationPlanner(config)
        self.human_detector = HumanDetector(config)
        self.safety_monitor = SafetyMonitor(config)
        
        # World model
        self.world_model = WorldModel()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize sensors
        self.initialize_sensors()
    
    def initialize_sensors(self):
        """Initialize all sensors for humanoid robot"""
        try:
            # Initialize RealSense cameras
            self.sensors[SensorType.HEAD_CAMERA] = RealSenseCamera('head', self.config)
            self.sensors[SensorType.CHEST_CAMERA] = RealSenseCamera('chest', self.config)
            self.sensors[SensorType.LEFT_HAND_CAMERA] = RealSenseCamera('left_hand', self.config)
            self.sensors[SensorType.RIGHT_HAND_CAMERA] = RealSenseCamera('right_hand', self.config)
            
            # Initialize IMU
            self.sensors[SensorType.IMU] = IMUSensor(self.config)
            
            # Initialize joint encoders
            self.sensors[SensorType.JOINT_ENCODERS] = JointSensor(self.config)
            
            print("✅ All sensors initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing sensors: {e}")
    
    def start_sensor_streaming(self):
        """Start streaming from all sensors"""
        for sensor_type, sensor in self.sensors.items():
            if hasattr(sensor, 'start_streaming'):
                sensor.start_streaming()
                
                # Start sensor thread
                thread = threading.Thread(
                    target=self._sensor_thread, 
                    args=(sensor_type, sensor)
                )
                thread.daemon = True
                thread.start()
                self.sensor_threads[sensor_type] = thread
        
        print("✅ All sensors started streaming")
    
    def _sensor_thread(self, sensor_type: SensorType, sensor):
        """Thread function for sensor data collection"""
        while True:
            try:
                # Get sensor data
                data = sensor.get_data()
                
                if data is not None:
                    # Create sensor data object
                    sensor_data = SensorData(
                        sensor_type=sensor_type,
                        timestamp=time.time(),
                        data=data,
                        confidence=1.0
                    )
                    
                    # Add to queue
                    self.sensor_data_queue.put(sensor_data)
                
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                print(f"❌ Error in sensor thread {sensor_type}: {e}")
                time.sleep(0.1)
    
    def process_perception_cycle(self) -> Dict:
        """Main perception processing cycle"""
        # Collect sensor data
        sensor_data = self.collect_sensor_data()
        
        # Update world model
        self.update_world_model(sensor_data)
        
        # Process perception modules
        perception_results = {
            'balance': self.balance_controller.update(self.world_model),
            'manipulation': self.manipulation_planner.plan(self.world_model),
            'human_interaction': self.human_detector.detect(self.world_model),
            'safety': self.safety_monitor.check_safety(self.world_model)
        }
        
        return perception_results
    
    def collect_sensor_data(self) -> Dict:
        """Collect data from all sensors"""
        sensor_data = {}
        
        # Process sensor data queue
        while not self.sensor_data_queue.empty():
            try:
                data = self.sensor_data_queue.get_nowait()
                sensor_data[data.sensor_type] = data
            except queue.Empty:
                break
        
        return sensor_data
    
    def update_world_model(self, sensor_data: Dict):
        """Update world model with sensor data"""
        # Update camera data
        for sensor_type in [SensorType.HEAD_CAMERA, SensorType.CHEST_CAMERA, 
                           SensorType.LEFT_HAND_CAMERA, SensorType.RIGHT_HAND_CAMERA]:
            if sensor_type in sensor_data:
                self.world_model.update_camera_data(sensor_type, sensor_data[sensor_type])
        
        # Update IMU data
        if SensorType.IMU in sensor_data:
            self.world_model.update_imu_data(sensor_data[SensorType.IMU])
        
        # Update joint data
        if SensorType.JOINT_ENCODERS in sensor_data:
            self.world_model.update_joint_data(sensor_data[SensorType.JOINT_ENCODERS])
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'fps': fps,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time,
            'sensor_count': len(self.sensors),
            'world_model_size': self.world_model.get_size()
        }
    
    def stop(self):
        """Stop all sensors and threads"""
        for sensor in self.sensors.values():
            if hasattr(sensor, 'stop'):
                sensor.stop()
        
        print("✅ All sensors stopped")

class BalanceController:
    """Advanced balance control for humanoid robots"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_com = np.array([0, 0, 0.8])  # Target center of mass
        self.stability_threshold = config.get('stability_threshold', 0.1)
        
    def update(self, world_model) -> BalanceInfo:
        """Update balance control"""
        # Calculate current center of mass
        current_com = self.calculate_center_of_mass(world_model)
        
        # Calculate support polygon
        support_polygon = self.calculate_support_polygon(world_model)
        
        # Check stability
        stability_margin = self.check_stability(current_com, support_polygon)
        
        # Determine balance state
        if stability_margin > self.stability_threshold:
            state = BalanceState.STABLE
        elif stability_margin > 0.05:
            state = BalanceState.WARNING
        elif stability_margin > 0.02:
            state = BalanceState.CRITICAL
        else:
            state = BalanceState.FALLING
        
        # Generate corrections
        corrections = self.generate_balance_corrections(current_com, support_polygon)
        
        return BalanceInfo(
            state=state,
            center_of_mass=current_com,
            support_polygon=support_polygon,
            stability_margin=stability_margin,
            corrections=corrections
        )
    
    def calculate_center_of_mass(self, world_model) -> np.ndarray:
        """Calculate current center of mass"""
        # Simplified calculation - in real implementation, this would use
        # joint positions and body segment masses
        joint_positions = world_model.get_joint_positions()
        
        if len(joint_positions) == 0:
            return self.target_com
        
        # Calculate weighted average of joint positions
        com = np.mean(joint_positions, axis=0)
        return com
    
    def calculate_support_polygon(self, world_model) -> np.ndarray:
        """Calculate support polygon from foot contact points"""
        # Get foot contact points from depth data
        foot_contacts = world_model.get_foot_contacts()
        
        if len(foot_contacts) < 3:
            # Default support polygon
            return np.array([[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]])
        
        # Calculate convex hull
        try:
            hull = ConvexHull(foot_contacts)
            return foot_contacts[hull.vertices]
        except:
            return np.array([[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1], [-0.1, 0.1]])
    
    def check_stability(self, com: np.ndarray, support_polygon: np.ndarray) -> float:
        """Check stability margin"""
        # Project COM to ground plane
        com_2d = com[:2]
        
        # Calculate distance to support polygon boundary
        min_distance = float('inf')
        for i in range(len(support_polygon)):
            p1 = support_polygon[i]
            p2 = support_polygon[(i + 1) % len(support_polygon)]
            
            # Distance from point to line segment
            distance = self.point_to_line_distance(com_2d, p1, p2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def point_to_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate distance from point to line segment"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj_length = max(0, min(line_len, proj_length))
        
        proj = line_start + proj_length * line_unitvec
        return np.linalg.norm(point - proj)
    
    def generate_balance_corrections(self, com: np.ndarray, support_polygon: np.ndarray) -> np.ndarray:
        """Generate balance correction commands"""
        # Calculate desired COM position (center of support polygon)
        desired_com = np.mean(support_polygon, axis=0)
        desired_com = np.append(desired_com, com[2])  # Keep height
        
        # Calculate correction
        correction = desired_com - com
        
        # Limit correction magnitude
        max_correction = 0.1  # 10cm
        correction_norm = np.linalg.norm(correction)
        if correction_norm > max_correction:
            correction = correction / correction_norm * max_correction
        
        return correction

class ManipulationPlanner:
    """Advanced manipulation planning for humanoid robots"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.targets = []
        self.next_target_id = 0
        
    def plan(self, world_model) -> List[ManipulationTarget]:
        """Plan manipulation targets"""
        # Detect objects in the environment
        objects = self.detect_objects(world_model)
        
        # Plan manipulation for each object
        targets = []
        for obj in objects:
            target = self.plan_object_manipulation(obj, world_model)
            if target:
                targets.append(target)
        
        self.targets = targets
        return targets
    
    def detect_objects(self, world_model) -> List[Dict]:
        """Detect objects in the environment"""
        # Simplified object detection - in real implementation, this would use
        # advanced computer vision techniques
        objects = []
        
        # Get point cloud data
        point_cloud = world_model.get_point_cloud()
        
        if len(point_cloud.points) > 0:
            # Simple clustering to find objects
            clusters = self.cluster_points(point_cloud)
            
            for i, cluster in enumerate(clusters):
                if len(cluster) > 100:  # Minimum cluster size
                    objects.append({
                        'id': i,
                        'points': cluster,
                        'center': np.mean(cluster, axis=0),
                        'size': len(cluster)
                    })
        
        return objects
    
    def cluster_points(self, point_cloud) -> List[np.ndarray]:
        """Cluster points to find objects"""
        # Simplified clustering - in real implementation, this would use
        # advanced clustering algorithms like DBSCAN
        points = np.asarray(point_cloud.points)
        
        if len(points) == 0:
            return []
        
        # Simple spatial clustering
        clusters = []
        used_points = set()
        
        for i, point in enumerate(points):
            if i in used_points:
                continue
            
            cluster = [point]
            used_points.add(i)
            
            # Find nearby points
            for j, other_point in enumerate(points):
                if j in used_points:
                    continue
                
                distance = np.linalg.norm(point - other_point)
                if distance < 0.1:  # 10cm threshold
                    cluster.append(other_point)
                    used_points.add(j)
            
            if len(cluster) > 10:  # Minimum cluster size
                clusters.append(np.array(cluster))
        
        return clusters
    
    def plan_object_manipulation(self, obj: Dict, world_model) -> Optional[ManipulationTarget]:
        """Plan manipulation for a specific object"""
        # Calculate grasp points
        grasp_points = self.calculate_grasp_points(obj)
        
        if len(grasp_points) == 0:
            return None
        
        # Estimate object orientation
        orientation = self.estimate_object_orientation(obj)
        
        # Calculate confidence
        confidence = self.calculate_manipulation_confidence(obj, grasp_points)
        
        return ManipulationTarget(
            id=self.next_target_id,
            position=obj['center'],
            orientation=orientation,
            grasp_points=grasp_points,
            confidence=confidence,
            object_type='unknown'
        )
    
    def calculate_grasp_points(self, obj: Dict) -> List[np.ndarray]:
        """Calculate grasp points for object"""
        # Simplified grasp point calculation
        center = obj['center']
        grasp_points = []
        
        # Add grasp points around the object
        for angle in [0, 90, 180, 270]:
            angle_rad = np.radians(angle)
            grasp_point = center + 0.05 * np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
            grasp_points.append(grasp_point)
        
        return grasp_points
    
    def estimate_object_orientation(self, obj: Dict) -> np.ndarray:
        """Estimate object orientation"""
        # Simplified orientation estimation
        return np.eye(3)  # Identity matrix
    
    def calculate_manipulation_confidence(self, obj: Dict, grasp_points: List[np.ndarray]) -> float:
        """Calculate manipulation confidence"""
        # Base confidence on object size and grasp point quality
        size_confidence = min(obj['size'] / 1000, 1.0)
        grasp_confidence = min(len(grasp_points) / 4, 1.0)
        
        return (size_confidence + grasp_confidence) / 2.0

class HumanDetector:
    """Human detection and interaction system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def detect(self, world_model) -> Dict:
        """Detect humans and gestures"""
        # Get camera data
        camera_data = world_model.get_camera_data()
        
        if not camera_data:
            return {'humans': [], 'gestures': []}
        
        humans = []
        gestures = []
        
        # Process each camera
        for sensor_type, data in camera_data.items():
            if 'color_image' in data:
                # Detect hands
                hand_gestures = self.detect_hand_gestures(data['color_image'])
                gestures.extend(hand_gestures)
                
                # Detect pose
                human_pose = self.detect_human_pose(data['color_image'])
                if human_pose:
                    humans.append(human_pose)
        
        return {
            'humans': humans,
            'gestures': gestures
        }
    
    def detect_hand_gestures(self, image: np.ndarray) -> List[Dict]:
        """Detect hand gestures"""
        results = self.hands.process(image)
        gestures = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self.classify_gesture(hand_landmarks)
                gestures.append(gesture)
        
        return gestures
    
    def detect_human_pose(self, image: np.ndarray) -> Optional[Dict]:
        """Detect human pose"""
        results = self.pose.process(image)
        
        if results.pose_landmarks:
            return {
                'landmarks': results.pose_landmarks,
                'confidence': 0.8  # Simplified confidence
            }
        
        return None
    
    def classify_gesture(self, hand_landmarks) -> Dict:
        """Classify hand gesture"""
        # Simplified gesture classification
        fingers = []
        
        # Thumb
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for i in range(1, 5):
            if hand_landmarks.landmark[4*i+3].y < hand_landmarks.landmark[4*i+1].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Classify gesture
        if sum(fingers) == 0:
            gesture_type = "fist"
        elif sum(fingers) == 5:
            gesture_type = "open_hand"
        elif fingers[0] == 1 and sum(fingers[1:]) == 0:
            gesture_type = "thumbs_up"
        else:
            gesture_type = "partial_hand"
        
        return {
            'type': gesture_type,
            'fingers': fingers,
            'confidence': 0.8
        }

class SafetyMonitor:
    """Safety monitoring system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.safety_thresholds = config.get('safety_thresholds', {})
    
    def check_safety(self, world_model) -> Dict:
        """Check safety conditions"""
        safety_alerts = []
        
        # Check balance safety
        balance_info = world_model.get_balance_info()
        if balance_info and balance_info.state == BalanceState.FALLING:
            safety_alerts.append({
                'type': 'balance',
                'level': 'critical',
                'message': 'Robot is falling!'
            })
        
        # Check collision safety
        collision_risk = self.check_collision_risk(world_model)
        if collision_risk > 0.8:
            safety_alerts.append({
                'type': 'collision',
                'level': 'critical',
                'message': 'High collision risk detected!'
            })
        
        return {
            'alerts': safety_alerts,
            'overall_safety': 'safe' if len(safety_alerts) == 0 else 'warning'
        }
    
    def check_collision_risk(self, world_model) -> float:
        """Check collision risk"""
        # Simplified collision risk calculation
        return 0.1  # Low risk by default

class WorldModel:
    """World model for humanoid robot"""
    
    def __init__(self):
        self.camera_data = {}
        self.imu_data = None
        self.joint_data = None
        self.balance_info = None
        self.point_cloud = None
    
    def update_camera_data(self, sensor_type: SensorType, data: SensorData):
        """Update camera data"""
        self.camera_data[sensor_type] = data.data
    
    def update_imu_data(self, data: SensorData):
        """Update IMU data"""
        self.imu_data = data.data
    
    def update_joint_data(self, data: SensorData):
        """Update joint data"""
        self.joint_data = data.data
    
    def get_camera_data(self) -> Dict:
        """Get camera data"""
        return self.camera_data
    
    def get_joint_positions(self) -> List[np.ndarray]:
        """Get joint positions"""
        if self.joint_data and 'positions' in self.joint_data:
            return self.joint_data['positions']
        return []
    
    def get_foot_contacts(self) -> List[np.ndarray]:
        """Get foot contact points"""
        # Simplified foot contact detection
        return [np.array([0.1, 0.1]), np.array([0.1, -0.1]), 
                np.array([-0.1, -0.1]), np.array([-0.1, 0.1])]
    
    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get point cloud data"""
        if self.point_cloud is None:
            self.point_cloud = o3d.geometry.PointCloud()
        return self.point_cloud
    
    def get_balance_info(self) -> Optional[BalanceInfo]:
        """Get balance information"""
        return self.balance_info
    
    def get_size(self) -> int:
        """Get world model size"""
        return len(self.camera_data) + (1 if self.imu_data else 0) + (1 if self.joint_data else 0)

# Simplified sensor classes for demonstration
class RealSenseCamera:
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.pipeline = None
        self.is_streaming = False
    
    def start_streaming(self):
        self.is_streaming = True
    
    def get_data(self) -> Optional[Dict]:
        if self.is_streaming:
            return {
                'color_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'depth_image': np.random.randint(0, 5000, (480, 640), dtype=np.uint16)
            }
        return None
    
    def stop(self):
        self.is_streaming = False

class IMUSensor:
    def __init__(self, config: Dict):
        self.config = config
    
    def get_data(self) -> Optional[Dict]:
        return {
            'acceleration': np.random.randn(3),
            'angular_velocity': np.random.randn(3),
            'orientation': np.random.randn(4)
        }

class JointSensor:
    def __init__(self, config: Dict):
        self.config = config
    
    def get_data(self) -> Optional[Dict]:
        return {
            'positions': [np.random.randn(3) for _ in range(20)],
            'velocities': [np.random.randn(3) for _ in range(20)]
        }

def main():
    """Main function for humanoid perception demo"""
    print("🤖 RealSense Humanoid Robot Perception System")
    print("=" * 50)
    
    # Configuration
    config = {
        'stability_threshold': 0.1,
        'safety_thresholds': {
            'collision_risk': 0.8,
            'balance_margin': 0.05
        }
    }
    
    # Initialize perception system
    perception_system = HumanoidPerceptionSystem(config)
    
    # Start sensor streaming
    perception_system.start_sensor_streaming()
    
    print("\n🎯 Humanoid Perception System Started!")
    print("Press 'q' to quit, 's' to save data")
    
    try:
        while True:
            # Process perception cycle
            results = perception_system.process_perception_cycle()
            
            # Get performance stats
            stats = perception_system.get_performance_stats()
            
            # Print results
            if perception_system.frame_count % 30 == 0:
                print(f"📊 FPS: {stats['fps']:.1f}, "
                      f"Balance: {results['balance'].state.value}, "
                      f"Targets: {len(results['manipulation'])}, "
                      f"Humans: {len(results['human_interaction']['humans'])}")
            
            perception_system.frame_count += 1
            time.sleep(0.1)  # 10Hz
    
    except KeyboardInterrupt:
        print("\n⏹️ Humanoid perception system interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        perception_system.stop()
        
        # Print final statistics
        stats = perception_system.get_performance_stats()
        print(f"\n📊 Perception system completed:")
        print(f"   Total frames: {stats['frame_count']}")
        print(f"   Average FPS: {stats['fps']:.1f}")
        print(f"   Sensors: {stats['sensor_count']}")
        print("✅ Humanoid perception system finished successfully")

if __name__ == "__main__":
    main()
