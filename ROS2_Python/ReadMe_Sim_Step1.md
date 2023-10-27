#  Simulation - Step 1 (Gazebo)

## Install Simulation Package
```shell
$ mkdir -p ~/turtlebot3_ws/src/
$ cd ~/turtlebot3_ws/src/
$ git clone -b foxy-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone -b foxy-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
$ git clone -b foxy-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ cd ~/turtlebot3_ws && colcon build --symlink-install
```

## Launch Simulation World
- Empty World
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=burger
$ ros2 launch turtlebot3_gazebo empty_world.launch.py
```

- TurtleBot3 World
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=waffle
$ ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

- TurtleBot3 House
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=waffle_pi
$ ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

## Operate TurtleBot3
```shell
$ source ~/turtlebot3_ws/install/setup.bash
$ export TURTLEBOT3_MODEL=waffle_pi
$ ros2 run turtlebot3_teleop teleop_keyboard
```
