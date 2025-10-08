#!/usr/bin/env python3
"""
RealSense Obstacle Detection System
==================================

This script implements a comprehensive obstacle detection system using RealSense cameras.
It detects obstacles in real-time, classifies them, and provides safety alerts.

Features:
- Real-time obstacle detection
- Obstacle classification and tracking
- Safety zone management
- Distance measurement and alerts
- Visual feedback and logging

Author: RealSense University
Level: 2 (Intermediate)
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ObstacleType(Enum):
    """Obstacle classification types"""
    UNKNOWN = "unknown"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    STATIC = "static"
    DYNAMIC = "dynamic"

class SafetyLevel(Enum):
    """Safety alert levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Obstacle:
    """Obstacle data structure"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    area: float
    distance: float
    obstacle_type: ObstacleType
    confidence: float
    timestamp: float

@dataclass
class SafetyZone:
    """Safety zone data structure"""
    level: SafetyLevel
    distance_range: Tuple[float, float]  # (min, max) in mm
    color: Tuple[int, int, int]
    alert_message: str

class ObstacleDetector:
    """Main obstacle detection system"""
    
    def __init__(self, config: Dict):
        """
        Initialize obstacle detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_obstacle_area = config.get('min_obstacle_area', 1000)
        self.max_obstacle_distance = config.get('max_obstacle_distance', 3000)
        self.min_obstacle_distance = config.get('min_obstacle_distance', 200)
        
        # Safety zones
        self.safety_zones = self._create_safety_zones()
        
        # Obstacle tracking
        self.obstacles = []
        self.next_obstacle_id = 0
        self.obstacle_history = []
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
    
    def _create_safety_zones(self) -> List[SafetyZone]:
        """Create safety zones based on configuration"""
        return [
            SafetyZone(
                level=SafetyLevel.CRITICAL,
                distance_range=(0, self.config.get('critical_distance', 500)),
                color=(0, 0, 255),  # Red
                alert_message="CRITICAL: Obstacle too close!"
            ),
            SafetyZone(
                level=SafetyLevel.WARNING,
                distance_range=(self.config.get('critical_distance', 500), 
                              self.config.get('warning_distance', 1000)),
                color=(0, 255, 255),  # Yellow
                alert_message="WARNING: Obstacle approaching"
            ),
            SafetyZone(
                level=SafetyLevel.SAFE,
                distance_range=(self.config.get('warning_distance', 1000), 
                              self.config.get('safe_distance', 2000)),
                color=(0, 255, 0),  # Green
                alert_message="SAFE: No immediate threat"
            )
        ]
    
    def detect_obstacles(self, depth_image: np.ndarray) -> List[Obstacle]:
        """
        Detect obstacles in depth image
        
        Args:
            depth_image: Input depth image
            
        Returns:
            List of detected obstacles
        """
        # Create obstacle mask
        obstacle_mask = self._create_obstacle_mask(depth_image)
        
        # Find obstacle contours
        contours = self._find_obstacle_contours(obstacle_mask)
        
        # Process contours into obstacles
        obstacles = []
        for contour in contours:
            obstacle = self._process_contour(contour, depth_image)
            if obstacle:
                obstacles.append(obstacle)
        
        # Update obstacle tracking
        self._update_obstacle_tracking(obstacles)
        
        return self.obstacles
    
    def _create_obstacle_mask(self, depth_image: np.ndarray) -> np.ndarray:
        """Create mask for potential obstacles"""
        # Filter valid depth range
        valid_depth = (depth_image > self.min_obstacle_distance) & \
                     (depth_image < self.max_obstacle_distance) & \
                     (depth_image > 0)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.morphologyEx(valid_depth.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        
        return obstacle_mask.astype(bool)
    
    def _find_obstacle_contours(self, obstacle_mask: np.ndarray) -> List:
        """Find obstacle contours"""
        contours, _ = cv2.findContours(
            obstacle_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_obstacle_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def _process_contour(self, contour, depth_image: np.ndarray) -> Optional[Obstacle]:
        """Process contour into obstacle object"""
        try:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Estimate distance
            distance = self._estimate_distance(contour, depth_image)
            
            # Classify obstacle
            obstacle_type = self._classify_obstacle(area, distance)
            
            # Calculate confidence
            confidence = self._calculate_confidence(contour, area, distance)
            
            return Obstacle(
                id=self.next_obstacle_id,
                bbox=(x, y, w, h),
                center=(center_x, center_y),
                area=area,
                distance=distance,
                obstacle_type=obstacle_type,
                confidence=confidence,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"❌ Error processing contour: {e}")
            return None
    
    def _estimate_distance(self, contour, depth_image: np.ndarray) -> float:
        """Estimate distance to obstacle"""
        # Get points within contour
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Calculate mean distance
        distances = depth_image[mask > 0]
        valid_distances = distances[distances > 0]
        
        if len(valid_distances) > 0:
            return np.mean(valid_distances)
        else:
            return 0.0
    
    def _classify_obstacle(self, area: float, distance: float) -> ObstacleType:
        """Classify obstacle based on properties"""
        if area > 5000 and distance < 1000:
            return ObstacleType.LARGE
        elif area < 2000:
            return ObstacleType.SMALL
        else:
            return ObstacleType.MEDIUM
    
    def _calculate_confidence(self, contour, area: float, distance: float) -> float:
        """Calculate detection confidence"""
        # Base confidence on area and distance
        area_confidence = min(area / 5000, 1.0)
        distance_confidence = 1.0 - (distance / self.max_obstacle_distance)
        
        # Combine confidences
        confidence = (area_confidence + distance_confidence) / 2.0
        
        return min(max(confidence, 0.0), 1.0)
    
    def _update_obstacle_tracking(self, new_obstacles: List[Obstacle]):
        """Update obstacle tracking"""
        # Simple tracking: assign new IDs to new obstacles
        for obstacle in new_obstacles:
            obstacle.id = self.next_obstacle_id
            self.next_obstacle_id += 1
        
        # Update obstacle list
        self.obstacles = new_obstacles
        
        # Add to history
        self.obstacle_history.append({
            'timestamp': time.time(),
            'obstacles': [obstacle.__dict__ for obstacle in new_obstacles]
        })
        
        # Keep only recent history
        if len(self.obstacle_history) > 100:
            self.obstacle_history = self.obstacle_history[-100:]
    
    def get_safety_status(self) -> Dict:
        """Get current safety status"""
        if not self.obstacles:
            return {
                'level': SafetyLevel.SAFE,
                'message': 'No obstacles detected',
                'obstacles': []
            }
        
        # Find closest obstacle
        closest_obstacle = min(self.obstacles, key=lambda x: x.distance)
        
        # Determine safety level
        for zone in self.safety_zones:
            if zone.distance_range[0] <= closest_obstacle.distance < zone.distance_range[1]:
                return {
                    'level': zone.level,
                    'message': zone.alert_message,
                    'obstacles': self.obstacles,
                    'closest_obstacle': closest_obstacle
                }
        
        return {
            'level': SafetyLevel.SAFE,
            'message': 'All obstacles are far away',
            'obstacles': self.obstacles
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'fps': fps,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time,
            'obstacle_count': len(self.obstacles)
        }

class ObstacleVisualizer:
    """Visualization for obstacle detection results"""
    
    def __init__(self):
        self.colors = {
            ObstacleType.SMALL: (255, 0, 0),    # Blue
            ObstacleType.MEDIUM: (0, 255, 0),   # Green
            ObstacleType.LARGE: (0, 0, 255),    # Red
            ObstacleType.UNKNOWN: (128, 128, 128)  # Gray
        }
    
    def draw_obstacles(self, image: np.ndarray, obstacles: List[Obstacle]) -> np.ndarray:
        """Draw obstacles on image"""
        result_image = image.copy()
        
        for obstacle in obstacles:
            # Draw bounding box
            x, y, w, h = obstacle.bbox
            color = self.colors.get(obstacle.obstacle_type, (128, 128, 128))
            
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center_x, center_y = obstacle.center
            cv2.circle(result_image, (center_x, center_y), 5, color, -1)
            
            # Draw obstacle info
            info_text = f"ID:{obstacle.id} {obstacle.distance:.0f}mm"
            cv2.putText(result_image, info_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_image
    
    def draw_safety_zones(self, image: np.ndarray, safety_zones: List[SafetyZone]) -> np.ndarray:
        """Draw safety zones on image"""
        result_image = image.copy()
        
        # Draw safety zone indicators
        for i, zone in enumerate(safety_zones):
            y_pos = 30 + i * 25
            cv2.rectangle(result_image, (10, y_pos - 15), (200, y_pos + 5), zone.color, -1)
            cv2.putText(result_image, zone.alert_message, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def draw_safety_status(self, image: np.ndarray, safety_status: Dict) -> np.ndarray:
        """Draw safety status on image"""
        result_image = image.copy()
        
        # Draw safety status
        level = safety_status['level']
        message = safety_status['message']
        
        # Choose color based on safety level
        if level == SafetyLevel.CRITICAL:
            color = (0, 0, 255)  # Red
        elif level == SafetyLevel.WARNING:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green
        
        # Draw status background
        cv2.rectangle(result_image, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.rectangle(result_image, (10, 10), (400, 50), color, 2)
        
        # Draw status text
        cv2.putText(result_image, f"SAFETY: {message}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result_image

class RealSenseCamera:
    """RealSense camera handler for obstacle detection"""
    
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
    """Main function for obstacle detection demo"""
    print("🚧 RealSense Obstacle Detection System")
    print("=" * 40)
    
    # Configuration
    config = {
        'min_obstacle_area': 1000,
        'max_obstacle_distance': 3000,
        'min_obstacle_distance': 200,
        'critical_distance': 500,
        'warning_distance': 1000,
        'safe_distance': 2000
    }
    
    # Initialize components
    camera = RealSenseCamera()
    detector = ObstacleDetector(config)
    visualizer = ObstacleVisualizer()
    
    if not camera.initialize():
        print("❌ Failed to initialize camera. Exiting.")
        return
    
    print("\n🎯 Obstacle Detection Started!")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            # Get frames
            depth_image, color_image = camera.get_frames()
            
            if depth_image is None or color_image is None:
                print("❌ Failed to get frames")
                break
            
            # Detect obstacles
            obstacles = detector.detect_obstacles(depth_image)
            
            # Get safety status
            safety_status = detector.get_safety_status()
            
            # Visualize results
            color_with_obstacles = visualizer.draw_obstacles(color_image, obstacles)
            color_with_safety = visualizer.draw_safety_status(color_with_obstacles, safety_status)
            color_with_zones = visualizer.draw_safety_zones(color_with_safety, detector.safety_zones)
            
            # Create depth visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Display images
            cv2.imshow('Obstacle Detection', color_with_zones)
            cv2.imshow('Depth Map', depth_colormap)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f'obstacle_detection_{timestamp}.png', color_with_zones)
                print(f"✅ Frame saved as obstacle_detection_{timestamp}.png")
            
            # Update frame count
            detector.frame_count += 1
            
            # Print status every 30 frames
            if detector.frame_count % 30 == 0:
                stats = detector.get_performance_stats()
                print(f"📊 FPS: {stats['fps']:.1f}, Obstacles: {len(obstacles)}, "
                      f"Safety: {safety_status['level'].value}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Obstacle detection interrupted by user")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = detector.get_performance_stats()
        print(f"\n📊 Detection completed:")
        print(f"   Total frames: {stats['frame_count']}")
        print(f"   Average FPS: {stats['fps']:.1f}")
        print(f"   Total time: {stats['elapsed_time']:.1f}s")
        print("✅ Obstacle detection finished successfully")

if __name__ == "__main__":
    main()
