# Module 3: Depth-Based Applications

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Build obstacle detection and distance alert systems
- Implement background segmentation using depth masks
- Create real-time object tracking applications
- Develop gesture recognition systems
- Build 3D object detection pipelines

## 🚧 Obstacle Detection System

### Basic Obstacle Detection

```python
import pyrealsense2 as rs
import numpy as np
import cv2

class ObstacleDetector:
    def __init__(self, min_distance=500, max_distance=2000):
        self.min_distance = min_distance  # mm
        self.max_distance = max_distance  # mm
        
    def detect_obstacles(self, depth_image):
        """Detect obstacles in depth image"""
        # Create mask for valid depth range
        obstacle_mask = (depth_image > self.min_distance) & (depth_image < self.max_distance)
        
        # Find contours
        contours, _ = cv2.findContours(
            obstacle_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum obstacle size
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
        
        return obstacles, obstacle_mask

# Usage example
detector = ObstacleDetector()
obstacles, mask = detector.detect_obstacles(depth_image)
```

### Advanced Obstacle Detection with Safety Zones

```python
class SafetyZoneDetector:
    def __init__(self, safety_distance=1000):
        self.safety_distance = safety_distance
        
    def create_safety_zones(self, depth_image, width, height):
        """Create safety zones based on depth"""
        # Define safety zones
        zones = {
            'critical': depth_image < self.safety_distance * 0.5,
            'warning': (depth_image >= self.safety_distance * 0.5) & (depth_image < self.safety_distance),
            'safe': depth_image >= self.safety_distance
        }
        
        return zones
    
    def get_zone_alerts(self, zones):
        """Get alerts based on safety zones"""
        alerts = []
        
        if np.any(zones['critical']):
            alerts.append({'type': 'critical', 'message': 'Obstacle too close!'})
        elif np.any(zones['warning']):
            alerts.append({'type': 'warning', 'message': 'Approaching obstacle'})
        
        return alerts
```

## 🎭 Background Segmentation

### Depth-Based Background Removal

```python
class BackgroundSegmenter:
    def __init__(self, background_distance=2000):
        self.background_distance = background_distance
        
    def segment_background(self, depth_image, color_image):
        """Remove background using depth information"""
        # Create foreground mask
        foreground_mask = depth_image < self.background_distance
        
        # Apply mask to color image
        segmented_image = color_image.copy()
        segmented_image[~foreground_mask] = [0, 0, 0]  # Black background
        
        return segmented_image, foreground_mask
    
    def adaptive_background_segmentation(self, depth_image, color_image, threshold=100):
        """Adaptive background segmentation"""
        # Calculate depth statistics
        mean_depth = np.mean(depth_image[depth_image > 0])
        std_depth = np.std(depth_image[depth_image > 0])
        
        # Create adaptive threshold
        adaptive_threshold = mean_depth + threshold
        
        # Create foreground mask
        foreground_mask = depth_image < adaptive_threshold
        
        # Apply mask
        segmented_image = color_image.copy()
        segmented_image[~foreground_mask] = [0, 0, 0]
        
        return segmented_image, foreground_mask
```

## 🎯 Object Tracking

### Real-Time Object Tracking

```python
import cv2
from collections import deque

class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        """Update object tracking"""
        if len(rects) == 0:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Calculate centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        # Match centroids to existing objects
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Calculate distances
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find minimum values
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Track matches
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])
        
        return self.objects
```

## 👋 Gesture Recognition

### Hand Gesture Detection

```python
class GestureRecognizer:
    def __init__(self):
        self.gesture_history = deque(maxlen=10)
        
    def detect_hand_gesture(self, depth_image, color_image):
        """Detect hand gestures using depth information"""
        # Segment hand region (simplified)
        hand_mask = self.segment_hand(depth_image)
        
        if np.sum(hand_mask) < 1000:  # No hand detected
            return None
        
        # Extract hand features
        features = self.extract_hand_features(hand_mask)
        
        # Classify gesture
        gesture = self.classify_gesture(features)
        
        return gesture
    
    def segment_hand(self, depth_image):
        """Segment hand region from depth image"""
        # Find closest objects (likely hands)
        min_depth = np.min(depth_image[depth_image > 0])
        hand_threshold = min_depth + 200  # 20cm range
        
        hand_mask = (depth_image > 0) & (depth_image < hand_threshold)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        hand_mask = cv2.morphologyEx(hand_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel)
        
        return hand_mask.astype(bool)
    
    def extract_hand_features(self, hand_mask):
        """Extract features from hand region"""
        # Find contours
        contours, _ = cv2.findContours(
            hand_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        area = cv2.contourArea(hand_contour)
        perimeter = cv2.arcLength(hand_contour, True)
        
        # Convex hull
        hull = cv2.convexHull(hand_contour)
        hull_area = cv2.contourArea(hull)
        
        # Features
        features = {
            'area': area,
            'perimeter': perimeter,
            'hull_area': hull_area,
            'solidity': area / hull_area if hull_area > 0 else 0,
            'aspect_ratio': self.calculate_aspect_ratio(hand_contour)
        }
        
        return features
    
    def classify_gesture(self, features):
        """Classify gesture based on features"""
        if not features:
            return "no_gesture"
        
        # Simple gesture classification
        if features['solidity'] > 0.8:
            return "fist"
        elif features['solidity'] < 0.6:
            return "open_hand"
        else:
            return "partial_hand"
    
    def calculate_aspect_ratio(self, contour):
        """Calculate aspect ratio of contour"""
        x, y, w, h = cv2.boundingRect(contour)
        return float(w) / h if h > 0 else 0
```

## 🧪 Hands-On Exercises

### Exercise 1: Obstacle Detection
1. Implement basic obstacle detection
2. Add safety zone visualization
3. Create distance-based alerts
4. Test with different objects

### Exercise 2: Background Segmentation
1. Create depth-based background removal
2. Implement adaptive segmentation
3. Compare with color-based methods
4. Optimize for real-time performance

### Exercise 3: Object Tracking
1. Implement object tracking system
2. Add trajectory visualization
3. Handle object appearance/disappearance
4. Test with multiple objects

### Exercise 4: Gesture Recognition
1. Build hand gesture detection
2. Implement gesture classification
3. Add gesture history tracking
4. Create gesture-based control interface

## 🎯 Next Steps

Ready to continue? → [Module 4: Cross-Platform Development](./module-4-cross-platform.md)
