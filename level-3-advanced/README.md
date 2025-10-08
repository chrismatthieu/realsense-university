# 🤖 Level 3: Advanced — AI + Robotics with RealSense

Welcome to Level 3 of RealSense University! This advanced level focuses on integrating RealSense depth perception with AI, SLAM, and embodied robotics systems.

## 🎯 Learning Objectives

By the end of Level 3, you will be able to:
- Implement Visual SLAM and 3D mapping systems
- Integrate multiple sensors for robust perception
- Build AI-powered perception pipelines
- Develop cloud robotics solutions
- Create autonomous navigation systems

## 📋 Prerequisites

- Completion of Level 2 or equivalent experience
- Advanced programming skills (Python, C++)
- Understanding of computer vision and AI concepts
- Familiarity with ROS2 and robotics frameworks
- Experience with machine learning libraries

## 📚 Modules Overview

| Module | Topic | Duration | Difficulty |
|--------|-------|----------|------------|
| [Module 1](./module-1-visual-slam.md) | Visual SLAM & Mapping | 120 min | ⭐⭐⭐⭐ |
| [Module 2](./module-2-sensor-fusion.md) | Sensor Fusion | 90 min | ⭐⭐⭐⭐ |
| [Module 3](./module-3-ai-perception.md) | AI Perception Pipelines | 150 min | ⭐⭐⭐⭐⭐ |
| [Module 4](./module-4-cloud-robotics.md) | Remote and Cloud Robotics | 90 min | ⭐⭐⭐⭐ |
| [Module 5](./module-5-mini-project.md) | Mini Project: Autonomous Navigation | 180 min | ⭐⭐⭐⭐⭐ |

## 🛠️ Required Hardware & Software

### Hardware
- **RealSense Camera**: D455 or D457 (IMU-enabled)
- **High-performance Computer**: GPU recommended
- **Additional Sensors**: LiDAR, IMU, wheel encoders
- **Robotics Platform**: Mobile robot or simulation environment

### Software
- **ROS2 Humble/Iron**: Latest version
- **OpenCV 4.5+**: Computer vision
- **Open3D**: 3D processing
- **PyTorch/TensorFlow**: AI frameworks
- **RTAB-Map/ORB-SLAM2**: SLAM libraries
- **Docker**: Containerization

## 🚀 Quick Start Guide

1. **Complete Level 2** or ensure advanced experience
2. **Set up development environment** with GPU support
3. **Install SLAM libraries** and AI frameworks
4. **Complete modules** in order for best learning experience
5. **Build the autonomous navigation project**

## 📖 Module Details

### [Module 1: Visual SLAM & Mapping](./module-1-visual-slam.md)
Master simultaneous localization and mapping with RealSense cameras.

**Key Topics:**
- RTAB-Map integration with RealSense
- ORB-SLAM2 for visual SLAM
- 3D map building and optimization
- Loop closure detection
- Pose estimation and tracking

### [Module 2: Sensor Fusion](./module-2-sensor-fusion.md)
Integrate multiple sensors for robust perception systems.

**Key Topics:**
- RealSense + LiDAR fusion
- IMU integration and calibration
- Multi-camera setups
- Data synchronization
- Kalman filtering and sensor fusion

### [Module 3: AI Perception Pipelines](./module-3-ai-perception.md)
Build AI-powered perception systems using RealSense data.

**Key Topics:**
- RGB-D object detection
- 3D semantic segmentation
- Gesture and pose estimation
- Real-time inference optimization
- Edge AI deployment

### [Module 4: Remote and Cloud Robotics](./module-4-cloud-robotics.md)
Develop cloud-based robotics solutions with RealSense.

**Key Topics:**
- ROS2 + Zenoh integration
- Remote data streaming
- Cloud inference services
- Edge-cloud collaboration
- Distributed robotics systems

### [Module 5: Mini Project: Autonomous Navigation](./module-5-mini-project.md)
Build a complete autonomous navigation system.

**Key Topics:**
- SLAM-based navigation
- Dynamic obstacle avoidance
- Path planning integration
- Multi-sensor fusion
- Real-time performance optimization

## 🧪 Hands-On Learning Approach

### Advanced Project-Based Learning
Each module includes sophisticated projects:
- **SLAM System**: Build a complete mapping solution
- **Sensor Fusion**: Create multi-sensor perception
- **AI Pipeline**: Develop intelligent perception
- **Cloud Robotics**: Deploy distributed systems
- **Final Project**: Complete autonomous navigation

### Real-World Applications
- **Autonomous Vehicles**: Navigation and perception
- **Robotic Manipulation**: 3D object understanding
- **Smart Cities**: Environmental monitoring
- **Healthcare Robotics**: Patient assistance
- **Industrial Automation**: Quality control and safety

## 🔧 Development Environment Setup

### GPU-Accelerated Setup

```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repository-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repository-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Open3D with CUDA
pip install open3d
```

### SLAM Libraries Installation

```bash
# Install RTAB-Map
sudo apt install ros-humble-rtabmap-ros

# Install ORB-SLAM2
git clone https://github.com/raulmur/ORB_SLAM2.git
cd ORB_SLAM2
chmod +x build.sh
./build.sh

# Install OpenVINO
pip install openvino
```

## 🆘 Getting Help

If you encounter issues:
1. Check the [troubleshooting section](./troubleshooting.md)
2. Review the [FAQ](./faq.md)
3. Join our [Discord community](https://discord.gg/realsense-university)
4. Search [GitHub issues](https://github.com/your-org/realsense-university/issues)
5. Consult [ROS2 documentation](https://docs.ros.org/en/humble/)

## 🎉 Completion Certificate

Upon completing all modules and the mini project, you'll receive a **Level 3 Completion Certificate** and be ready to advance to [Level 4: Expert](./../level-4-expert/).

## 🔗 Related Resources

- [RTAB-Map Documentation](http://wiki.ros.org/rtabmap_ros)
- [ORB-SLAM2 GitHub](https://github.com/raulmur/ORB_SLAM2)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Ready to start?** Let's begin with [Module 1: Visual SLAM & Mapping](./module-1-visual-slam.md)!
