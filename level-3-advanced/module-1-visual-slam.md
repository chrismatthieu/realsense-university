# Module 1: Visual SLAM & Mapping

## 🎯 Learning Objectives

By the end of this module, you will be able to:
- Implement Visual SLAM with RealSense cameras
- Build 3D maps using RTAB-Map
- Integrate ORB-SLAM2 for visual tracking
- Perform loop closure detection
- Optimize maps for navigation

## 🗺️ Understanding Visual SLAM

### What is Visual SLAM?

**Visual SLAM** (Simultaneous Localization and Mapping) is a technique that allows a robot to:
- **Localize**: Determine its position in an unknown environment
- **Map**: Build a map of the environment
- **Navigate**: Plan paths and avoid obstacles

### SLAM Components

```python
class SLAMSystem:
    def __init__(self):
        self.map = None
        self.pose = None
        self.trajectory = []
        self.keyframes = []
        
    def process_frame(self, rgb_image, depth_image):
        """Process a new frame for SLAM"""
        # 1. Feature extraction
        features = self.extract_features(rgb_image)
        
        # 2. Feature matching
        matches = self.match_features(features)
        
        # 3. Pose estimation
        pose = self.estimate_pose(matches, depth_image)
        
        # 4. Map update
        self.update_map(pose, depth_image)
        
        # 5. Loop closure detection
        self.detect_loop_closure()
        
        return pose
```

## 🛠️ RTAB-Map Integration

### RTAB-Map Setup

```bash
# Install RTAB-Map
sudo apt install ros-humble-rtabmap-ros

# Launch RTAB-Map with RealSense
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start" \
    depth_topic:=/camera/depth/image_rect_raw \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info
```

### Custom RTAB-Map Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import cv2
import numpy as np

class RTABMapNode(Node):
    def __init__(self):
        super().__init__('rtabmap_node')
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'slam/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, 'slam/map', 10)
        
        # SLAM state
        self.current_pose = None
        self.map_data = None
        
    def rgb_callback(self, msg):
        """Process RGB images for SLAM"""
        # Convert to OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Process for SLAM
        self.process_rgb_frame(rgb_image)
        
    def depth_callback(self, msg):
        """Process depth images for SLAM"""
        # Convert to OpenCV
        depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        
        # Process for SLAM
        self.process_depth_frame(depth_image)
        
    def process_rgb_frame(self, rgb_image):
        """Process RGB frame for feature extraction"""
        # Extract ORB features
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(rgb_image, None)
        
        # Store for SLAM processing
        self.current_features = {
            'keypoints': keypoints,
            'descriptors': descriptors
        }
        
    def process_depth_frame(self, depth_image):
        """Process depth frame for 3D mapping"""
        # Convert depth to 3D points
        points_3d = self.depth_to_3d(depth_image)
        
        # Update map
        self.update_3d_map(points_3d)
        
    def depth_to_3d(self, depth_image):
        """Convert depth image to 3D points"""
        height, width = depth_image.shape
        points_3d = []
        
        for y in range(height):
            for x in range(width):
                depth = depth_image[y, x]
                if depth > 0:
                    # Convert to 3D coordinates
                    z = depth / 1000.0  # Convert mm to m
                    x_3d = (x - width/2) * z / self.fx
                    y_3d = (y - height/2) * z / self.fy
                    points_3d.append([x_3d, y_3d, z])
        
        return np.array(points_3d)
```

## 🎯 ORB-SLAM2 Integration

### ORB-SLAM2 Setup

```bash
# Clone ORB-SLAM2
git clone https://github.com/raulmur/ORB_SLAM2.git
cd ORB_SLAM2

# Build
chmod +x build.sh
./build.sh

# Install Python bindings
pip install orb-slam2-python
```

### ORB-SLAM2 Python Interface

```python
import numpy as np
import cv2
from orb_slam2 import ORB_SLAM2

class ORBSLAM2Wrapper:
    def __init__(self, vocab_path, settings_path):
        self.slam = ORB_SLAM2(vocab_path, settings_path, True)
        
    def process_frame(self, rgb_image, depth_image):
        """Process frame with ORB-SLAM2"""
        # Convert images
        rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Process with ORB-SLAM2
        pose = self.slam.TrackRGBD(rgb_gray, depth_image)
        
        return pose
        
    def get_map_points(self):
        """Get 3D map points"""
        return self.slam.GetMapPoints()
        
    def get_keyframes(self):
        """Get keyframes"""
        return self.slam.GetKeyFrames()
```

## 🧪 Hands-On Exercises

### Exercise 1: Basic SLAM Setup
1. Install RTAB-Map and ORB-SLAM2
2. Launch SLAM with RealSense camera
3. Move camera around to build map
4. Visualize results in RViz

### Exercise 2: Custom SLAM Node
1. Create custom SLAM node
2. Implement feature extraction
3. Add pose estimation
4. Test with real camera data

### Exercise 3: Map Optimization
1. Implement loop closure detection
2. Optimize map using bundle adjustment
3. Compare before/after results
4. Measure mapping accuracy

## 🎯 Next Steps

Ready to continue? → [Module 2: Sensor Fusion](./module-2-sensor-fusion.md)
