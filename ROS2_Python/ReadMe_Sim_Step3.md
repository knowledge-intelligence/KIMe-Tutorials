# Simulation -  Step 3 (Navigation - Nav2) [ROS2 - Foxy]

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





## Not Tested
- https://www.youtube.com/watch?v=IrJmuow1r7g
- https://roboticsbackend.com/ros2-nav2-tutorial/
- First, as you may know, ROS2 communication is based on DDS (for the middleware). No need to dive into this now, you just have to know that there are several possible DDS implementations, and the default one for ROS2 is Fast DDS. Unfortunately it doesn’t work so well with Nav2, so it’s been recommended to use Cyclone DDS instead.
- $ sudo gedit /opt/ros/humble/share/turtlebot3_navigation2/param/burger.yaml
- #robot_model_type: "differential"
- robot_model_type: "nav2_amcl::DifferentialMotionModel"
