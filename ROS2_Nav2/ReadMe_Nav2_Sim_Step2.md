# Simulation[ROS2 - Humble] - Step 2 (Turtlebot3 Gazebo SLAM - Cartographer)

## Launch Simulation World
```bash
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## Run SLAM Node - Cartographer SLAM
```bash
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```

## Run Teleoperation Node
```bash
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard
```

## Save Map
```bash
source /opt/ros/jazzy/setup.bash
ros2 run nav2_map_server map_saver_cli -f ~/map
```
