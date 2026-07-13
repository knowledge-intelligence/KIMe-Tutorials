#  Simulation[ROS2 - Humble] - Step 1 (Turtlebot3 Gazebo Setting)

## ROS2 Package Check
```shell
ros2 pkg list | grep 'turtlebot*'
```

## Clone & Build Simulation Package
```shell
export ROS_DISTRO=jazzy
mkdir -p ~/turtlebot3_ws/src/
cd ~/turtlebot3_ws/src/
git clone -b $ROS_DISTRO https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b $ROS_DISTRO https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b $ROS_DISTRO https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

# Low Computer Resource
cd ~/turtlebot3_ws && colcon build --executor sequential
# General Computer Resource
cd ~/turtlebot3_ws && colcon build --symlink-install
```


## Launch Simulation World
- Empty World
```shell
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

- TurtleBot3 World
```shell
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

- TurtleBot3 House
```shell
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

## Tele-Operate TurtleBot3
```shell
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 run turtlebot3_teleop teleop_keyboard
```


## launch 관련 프로세스 통째로 확인 후 종료
```shell
pkill -9 -f "gz sim|ros2 launch"
```