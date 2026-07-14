# KIMe-Tutorials for ROS2 Tutorials

## Sourcing ROS2 Jazzy
```bash
source /opt/ros/jazzy/setup.bash
```

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Install VSCode
```bash
sudo snap install --classic code
```

## Install Gazebo
```bash
export ROS_DISTRO=jazzy
sudo apt update
sudo apt install ros-$ROS_DISTRO-ros-gz-sim ros-$ROS_DISTRO-ros-gz
```

## Install Nav2
```bash
export ROS_DISTRO=jazzy
sudo apt update
sudo apt install ros-$ROS_DISTRO-cartographer ros-$ROS_DISTRO-cartographer-ros ros-$ROS_DISTRO-navigation2 ros-$ROS_DISTRO-nav2-bringup ros-$ROS_DISTRO-rmw-cyclonedds-cpp
sudo apt install ros-$ROS_DISTRO-dynamixel-sdk
```