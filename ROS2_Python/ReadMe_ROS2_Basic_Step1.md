# Turtlesim


```bash
ros2 run turtlesim turtlesim_node
```

```bash
ros2 node info /turtlesim
```

```bash
ros2 topic list -t
```

```bash
rqt_graph
```

```bash
ros2 run turtlesim turtle_teleop_key
```

```bash
ros2 topic echo /turtle1/cmd_vel
```

```bash
ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```

```bash
ros2 topic pub –-rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
```

```bash
ros2 bag record /turtle1/cmd_vel
```

```bash
ros2 bag info rosbag2_2022_02_15-20_37_25/
```

```bash
ros2 bag play rosbag2_2022_02_15-20_37_25/
```
