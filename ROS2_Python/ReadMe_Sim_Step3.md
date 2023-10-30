# Simulation -  Step 3 (Navigation - Nav2)

## Launch Simulation World
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
$ export TURTLEBOT3_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## Run Navigation Node - w/ Navigation2
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
$ export TURTLEBOT3_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=$HOME/map.yaml
```


## Estimate Initial Pose
- Using "2D Pose Estimate" Button (Important !!!)

- Run Teleoperation Node (to precisely locate)
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
$ export TURTLEBOT3_MODEL=burger
$ ros2 run turtlebot3_teleop teleop_keyboard
```

- Move the robot back and forth (to precisely locate)
- Using "Ctrl + C" to terminate the keyboard teleoperation node


## Set Navigation Goal
- Click the "Navigation2 Goal" button
