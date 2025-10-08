# 💻 Code Examples & Templates

Welcome to the Code Examples section! This repository contains ready-to-use code examples, templates, and starter projects for all levels of RealSense University.

## 📁 Repository Structure

```
code-examples/
├── level-1-beginner/          # Beginner code examples
├── level-2-intermediate/      # Intermediate code examples
├── level-3-advanced/          # Advanced code examples
├── level-4-expert/            # Expert code examples
└── README.md                  # This file
```

## 🚀 Quick Start

### 📋 Prerequisites
- Python 3.7+
- RealSense SDK 2.0
- Required Python packages (see individual examples)

### 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/your-org/realsense-university.git
cd realsense-university/code-examples

# Install dependencies
pip install -r requirements.txt

# Run an example
python level-1-beginner/basic_camera_test.py
```

## 📚 Level-Specific Examples

### 🧭 Level 1: Beginner Examples
**Basic RealSense functionality and simple applications**

| Example | Description | Difficulty |
|---------|-------------|------------|
| [basic_camera_test.py](./level-1-beginner/basic_camera_test.py) | Test camera connection and basic functionality | ⭐ |
| [depth_visualization.py](./level-1-beginner/depth_visualization.py) | Advanced depth visualization with multiple colormaps | ⭐ |
| [point_cloud_generator.py](./level-1-beginner/point_cloud_generator.py) | Generate and visualize 3D point clouds | ⭐⭐ |
| [distance_measurement.py](./level-1-beginner/distance_measurement.py) | Interactive distance measurement tool | ⭐⭐ |

### ⚙️ Level 2: Intermediate Examples
**ROS2 integration and advanced applications**

| Example | Description | Difficulty |
|---------|-------------|------------|
| [ros2_basic_node.py](./level-2-intermediate/ros2_basic_node.py) | Complete ROS2 node with topic publishing | ⭐⭐ |
| [obstacle_detector.py](./level-2-intermediate/obstacle_detector.py) | Real-time obstacle detection and safety zones | ⭐⭐⭐ |

### 🤖 Level 3: Advanced Examples
**AI integration and complex robotics applications**

| Example | Description | Difficulty |
|---------|-------------|------------|
| [slam_system.py](./level-3-advanced/slam_system.py) | Complete Visual SLAM implementation | ⭐⭐⭐⭐ |

### 🧑‍🏫 Level 4: Expert Examples
**Advanced research and development projects**

| Example | Description | Difficulty |
|---------|-------------|------------|
| [humanoid_perception.py](./level-4-expert/humanoid_perception.py) | Advanced humanoid robot perception system | ⭐⭐⭐⭐⭐ |

## 🛠️ Common Utilities

### 📦 Utility Functions
- **camera_utils.py**: Camera management and configuration
- **depth_utils.py**: Depth image processing utilities
- **point_cloud_utils.py**: Point cloud processing functions
- **visualization_utils.py**: 3D visualization helpers
- **performance_utils.py**: Performance monitoring and optimization

### 🔧 Configuration Files
- **camera_config.yaml**: Camera configuration templates
- **ros2_config.yaml**: ROS2 parameter files
- **ai_model_config.yaml**: AI model configuration
- **deployment_config.yaml**: Deployment configuration

## 📖 Usage Examples

### 🧪 Basic Usage
```python
from level_1_beginner.basic_camera_test import RealSenseCamera

# Initialize camera
camera = RealSenseCamera()

# Start streaming
camera.start_streaming()

# Get frames
depth, color = camera.get_frames()

# Process frames
processed_depth = camera.process_depth(depth)

# Stop streaming
camera.stop_streaming()
```

### 🚀 Advanced Usage
```python
from level_3_advanced.slam_system import SLAMSystem
from level_3_advanced.sensor_fusion import SensorFusion

# Initialize SLAM system
slam = SLAMSystem()

# Initialize sensor fusion
fusion = SensorFusion()

# Process data
pose = slam.process_frame(rgb_image, depth_image)
fused_data = fusion.fuse_sensors(pose, imu_data, lidar_data)
```

## 🧪 Testing

### 🧪 Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific level tests
python -m pytest tests/level_1_beginner/
python -m pytest tests/level_2_intermediate/
python -m pytest tests/level_3_advanced/
python -m pytest tests/level_4_expert/
```

### 🔍 Code Quality
```bash
# Run linting
flake8 code-examples/

# Run type checking
mypy code-examples/

# Run formatting
black code-examples/
```

## 📚 Documentation

### 📖 API Documentation
- **Level 1**: [Beginner API Docs](./level-1-beginner/README.md)
- **Level 2**: [Intermediate API Docs](./level-2-intermediate/README.md)
- **Level 3**: [Advanced API Docs](./level-3-advanced/README.md)
- **Level 4**: [Expert API Docs](./level-4-expert/README.md)

### 🎯 Tutorials
- **Getting Started**: [Basic Tutorial](./tutorials/getting-started.md)
- **ROS2 Integration**: [ROS2 Tutorial](./tutorials/ros2-integration.md)
- **AI Integration**: [AI Tutorial](./tutorials/ai-integration.md)
- **Deployment**: [Deployment Tutorial](./tutorials/deployment.md)

## 🤝 Contributing

### 📝 How to Contribute
1. **Fork the repository**
2. **Create a feature branch**
3. **Add your code example**
4. **Write tests and documentation**
5. **Submit a pull request**

### 📋 Contribution Guidelines
- **Code Style**: Follow PEP 8 and use Black formatting
- **Documentation**: Include docstrings and comments
- **Testing**: Write unit tests for your code
- **Examples**: Provide clear usage examples

### 🏆 Recognition
- **Contributors**: Listed in CONTRIBUTORS.md
- **Badges**: Earn contributor badges
- **Recognition**: Featured in community highlights

## 📞 Support

### 🆘 Getting Help
- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/realsense-university/issues)
- **Discord**: [Join our community](https://discord.gg/realsense-university)
- **Forum**: [Ask questions](https://forum.realsense-university.com)
- **Email**: [Contact support](mailto:support@realsense-university.com)

### 📚 Resources
- **Documentation**: [Official docs](https://docs.realsense-university.com)
- **Video Tutorials**: [YouTube channel](https://youtube.com/@realsense-university)
- **Community**: [Join our community](https://discord.gg/realsense-university)

---

**Ready to start coding?** Choose your level and explore the examples!
