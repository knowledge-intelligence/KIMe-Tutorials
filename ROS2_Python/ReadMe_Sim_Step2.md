# Simulation - Step 2 (SLAM - Cartographer)  [ROS2 - Foxy]

## Launch Simulation World
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## Run SLAM Node - Cartographer SLAM
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=burger
$ ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```

## Run Teleoperation Node
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=burger
$ ros2 run turtlebot3_teleop teleop_keyboard
```

## Save Map
```shell
$ ros2 run nav2_map_server map_saver_cli -f ~/map
```
