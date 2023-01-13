# ROS1 DRL Navigation Tutorial

## Clone Code
git clone https://github.com/reiniscimurs/DRL-robot-navigation


## ROS1 Build
cd ~/DRL-robot-navigation/catkin_ws
catkin_make_isolated


## Install Gazebo
curl -sSL http://get.gazebosim.org | sh


## Install Pytorch 1.10 - GPU
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

## Install Pytorch 1.10 - CPU
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

## Install Tensorboard
pip3 install tensorboard

## Install etc.
pip3 install squaternion


## ROS1 & Gazebo Setting
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
source ~/.bashrc
cd ~/DRL-robot-navigation/catkin_ws
source devel_isolated/setup.bash


## Run Code
cd ~/DRL-robot-navigation/TD3
python3 train_velodyne_td3.py

## Tensorboard
cd ~/DRL-robot-navigation/TD3
tensorboard --logdir 'runs'



## Run Gazebo Client to see the simulation
source /usr/share/gazebo/setup.sh
gzclient
% (or Check roslaunch setting to see Gazebo simulation client)

## Check Gazebo Run
gazebo --verbose

## Kill Gazebo
killall gzserver
killall rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
