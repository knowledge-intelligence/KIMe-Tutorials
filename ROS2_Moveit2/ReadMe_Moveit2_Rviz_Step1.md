#  Moveit2 RViz [ROS2 - Humble] - Step 1 (Panda)

## Reference
- https://moveit.picknik.ai/main/doc/tutorials/quickstart_in_rviz/quickstart_in_rviz_tutorial.html
- https://moveit.picknik.ai/main/doc/examples/setup_assistant/setup_assistant_tutorial.html
- https://github.com/moveit/moveit2_tutorials/tree/0b64862bae30bcf9c3b7ff5435c4749538123d57/doc/tutorials/quickstart_in_rviz
- https://github.com/Kinovarobotics/ros2_kortex/blob/97a0e7c9a2b7970f8de5830919e2fe0d7eea3bf6/kortex_moveit_config/kinova_gen3_7dof_robotiq_2f_85_moveit_config/package.xml#L4



## Make Workspace & Download Panda Resource
```shell
mkdir -p ~/ws_moveit2/src
cd ~/ws_moveit2/src
git clone --branch humble https://github.com/moveit/moveit_resources.git
```
## Run Moveit Setup Assistant for "panda_moveit_config"
```shell
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```
Tutorial(Humble) - MoveIt Setup Assistant.pdf

## Build
```shell
cd ~/ws_moveit2
colcon build --packages-select moveit_resources_panda_moveit_config
```

## Run Launch (Panda Moveit2 RViz)
```shell
ros2 launch moveit_resource_panda_moveit_config demo.launch.py
```