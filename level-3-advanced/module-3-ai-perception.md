# Module 3: AI Perception Pipelines

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Build RGB-D object detection systems
- Implement 3D semantic segmentation
- Create gesture and pose estimation pipelines
- Optimize AI models for real-time inference
- Deploy AI models on edge devices

## 🤖 RGB-D Object Detection

### YOLO with Depth Integration

```python
import torch
import cv2
import numpy as np
from ultralytics import YOLO

class RGBDObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.depth_scale = 1000.0  # Convert mm to m
        
    def detect_objects(self, rgb_image, depth_image):
        """Detect objects with depth information"""
        # Run YOLO detection
        results = self.model(rgb_image)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get 2D bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate depth
                    depth = self.calculate_object_depth(
                        depth_image, int(x1), int(y1), int(x2), int(y2)
                    )
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'depth': depth,
                        'class_name': self.model.names[class_id]
                    }
                    detections.append(detection)
        
        return detections
    
    def calculate_object_depth(self, depth_image, x1, y1, x2, y2):
        """Calculate average depth of object"""
        # Extract depth region
        depth_region = depth_image[y1:y2, x1:x2]
        
        # Filter valid depths
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) > 0:
            return np.mean(valid_depths) / self.depth_scale
        else:
            return 0.0
```

### 3D Bounding Box Estimation

```python
class BoundingBox3D:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
    def estimate_3d_bbox(self, detection, depth_image):
        """Estimate 3D bounding box from 2D detection"""
        x1, y1, x2, y2 = detection['bbox']
        depth = detection['depth']
        
        # Calculate 3D corners
        corners_3d = []
        
        for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            # Convert to 3D
            z = depth
            x_3d = (x - self.cx) * z / self.fx
            y_3d = (y - self.cy) * z / self.fy
            
            corners_3d.append([x_3d, y_3d, z])
        
        # Estimate object dimensions
        width = abs(corners_3d[1][0] - corners_3d[0][0])
        height = abs(corners_3d[2][1] - corners_3d[0][1])
        
        # Assume depth based on object class
        depth_estimate = self.estimate_object_depth(detection['class_name'])
        
        return {
            'center': np.mean(corners_3d, axis=0),
            'dimensions': [width, height, depth_estimate],
            'corners': corners_3d
        }
    
    def estimate_object_depth(self, class_name):
        """Estimate object depth based on class"""
        depth_estimates = {
            'person': 0.3,
            'car': 1.5,
            'bottle': 0.1,
            'chair': 0.5,
            'table': 0.7
        }
        return depth_estimates.get(class_name, 0.5)
```

## 🎨 3D Semantic Segmentation

### Point Cloud Segmentation

```python
import open3d as o3d
from sklearn.cluster import DBSCAN

class PointCloudSegmenter:
    def __init__(self):
        self.segmenter = DBSCAN(eps=0.02, min_samples=10)
        
    def segment_point_cloud(self, point_cloud, colors=None):
        """Segment point cloud into objects"""
        # Convert to numpy array
        points = np.asarray(point_cloud.points)
        
        # Perform clustering
        labels = self.segmenter.fit_predict(points)
        
        # Create segmented point clouds
        segments = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise
                continue
                
            # Create segment
            segment_mask = labels == label
            segment_points = points[segment_mask]
            
            if len(segment_points) > 100:  # Minimum segment size
                segment_pcd = o3d.geometry.PointCloud()
                segment_pcd.points = o3d.utility.Vector3dVector(segment_points)
                
                if colors is not None:
                    segment_colors = colors[segment_mask]
                    segment_pcd.colors = o3d.utility.Vector3dVector(segment_colors)
                
                segments.append({
                    'point_cloud': segment_pcd,
                    'label': label,
                    'size': len(segment_points)
                })
        
        return segments
```

### Semantic Segmentation with Deep Learning

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class RGBDSemanticSegmentation:
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def segment_image(self, rgb_image, depth_image):
        """Perform semantic segmentation"""
        # Preprocess images
        rgb_tensor = self.transform(rgb_image).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
        
        # Normalize depth
        depth_tensor = depth_tensor / 1000.0  # Convert mm to m
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(rgb_tensor, depth_tensor)
            segmentation = torch.argmax(predictions, dim=1)
        
        return segmentation.squeeze().cpu().numpy()
```

## 👋 Gesture and Pose Estimation

### Hand Gesture Recognition

```python
import mediapipe as mp

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_gestures(self, rgb_image, depth_image):
        """Detect hand gestures"""
        # Process with MediaPipe
        results = self.hands.process(rgb_image)
        
        gestures = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand features
                features = self.extract_hand_features(hand_landmarks)
                
                # Classify gesture
                gesture = self.classify_gesture(features)
                
                # Get 3D position from depth
                position_3d = self.get_hand_position_3d(hand_landmarks, depth_image)
                
                gestures.append({
                    'gesture': gesture,
                    'position_3d': position_3d,
                    'landmarks': hand_landmarks
                })
        
        return gestures
    
    def extract_hand_features(self, landmarks):
        """Extract features from hand landmarks"""
        # Calculate finger states
        fingers = []
        
        # Thumb
        if landmarks.landmark[4].x > landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for i in range(1, 5):
            if landmarks.landmark[4*i+3].y < landmarks.landmark[4*i+1].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def classify_gesture(self, features):
        """Classify gesture based on features"""
        gesture_map = {
            (0, 0, 0, 0, 0): "fist",
            (1, 1, 1, 1, 1): "open_hand",
            (0, 1, 1, 1, 1): "thumbs_down",
            (1, 0, 0, 0, 0): "thumbs_up",
            (0, 1, 0, 0, 0): "pointing"
        }
        
        return gesture_map.get(tuple(features), "unknown")
    
    def get_hand_position_3d(self, landmarks, depth_image):
        """Get 3D position of hand center"""
        # Get center of hand
        center_x = int(landmarks.landmark[9].x * depth_image.shape[1])
        center_y = int(landmarks.landmark[9].y * depth_image.shape[0])
        
        # Get depth
        depth = depth_image[center_y, center_x]
        
        if depth > 0:
            # Convert to 3D coordinates
            fx, fy = 525, 525  # Camera intrinsics
            cx, cy = depth_image.shape[1]//2, depth_image.shape[0]//2
            
            z = depth / 1000.0
            x = (center_x - cx) * z / fx
            y = (center_y - cy) * z / fy
            
            return [x, y, z]
        
        return [0, 0, 0]
```

## ⚡ Real-Time Optimization

### Model Optimization

```python
import torch
import torch.quantization as quantization

class ModelOptimizer:
    def __init__(self, model):
        self.model = model
        
    def optimize_for_inference(self):
        """Optimize model for real-time inference"""
        # Set to evaluation mode
        self.model.eval()
        
        # Apply optimizations
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.optimize_for_inference(self.model)
        
        return self.model
    
    def quantize_model(self):
        """Quantize model for faster inference"""
        # Prepare for quantization
        self.model.eval()
        self.model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Apply quantization
        quantized_model = quantization.quantize_dynamic(
            self.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def optimize_for_mobile(self):
        """Optimize for mobile deployment"""
        # Convert to TorchScript
        scripted_model = torch.jit.script(self.model)
        
        # Optimize for mobile
        mobile_model = torch.jit.optimize_for_mobile(scripted_model)
        
        return mobile_model
```

### Performance Monitoring

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        
    def start_frame(self):
        """Start timing a frame"""
        self.frame_start = time.time()
        
    def end_frame(self):
        """End timing a frame"""
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        
        # Record system metrics
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
        
        # GPU usage if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usage.append(gpus[0].load * 100)
        except:
            pass
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.frame_times:
            return None
        
        return {
            'avg_fps': 1.0 / np.mean(self.frame_times),
            'avg_frame_time': np.mean(self.frame_times),
            'avg_cpu_usage': np.mean(self.cpu_usage),
            'avg_memory_usage': np.mean(self.memory_usage),
            'avg_gpu_usage': np.mean(self.gpu_usage) if self.gpu_usage else 0
        }
```

## 🧪 Hands-On Exercises

### Exercise 1: RGB-D Object Detection
1. Implement YOLO with depth integration
2. Test with different objects
3. Compare 2D vs 3D detection accuracy
4. Optimize for real-time performance

### Exercise 2: 3D Semantic Segmentation
1. Build point cloud segmentation system
2. Implement deep learning segmentation
3. Compare different segmentation methods
4. Visualize results in 3D

### Exercise 3: Gesture Recognition
1. Implement hand gesture detection
2. Add 3D position tracking
3. Create gesture-based control interface
4. Test with different users

### Exercise 4: Performance Optimization
1. Optimize AI models for inference
2. Implement performance monitoring
3. Compare different optimization techniques
4. Deploy on edge devices

## 🎯 Next Steps

Ready to continue? → [Module 4: Remote and Cloud Robotics](./module-4-cloud-robotics.md)
