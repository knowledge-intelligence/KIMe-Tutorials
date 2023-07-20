# ROS1 DRL Navigation Tutorial

## Clone Code
git clone https://github.com/reiniscimurs/DRL-robot-navigation


## ROS1 Build
cd ~/DRL-robot-navigation/catkin_ws <br>
catkin_make_isolated


## Install Gazebo
(if needed) sudo apt  install curl <br>
curl -sSL http://get.gazebosim.org | sh

## Install Pytorch
(if needed) sudo apt install python3-pip

### Install Pytorch 1.10 - GPU
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

### Install Pytorch 1.10 - CPU
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

## Install Tensorboard
pip3 install tensorboard

## Install etc.
pip3 install squaternion


## ROS1 & Gazebo Setting
export ROS_HOSTNAME=localhost <br>
export ROS_MASTER_URI=http://localhost:11311 <br>
export ROS_PORT_SIM=11311 <br>
export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch <br>
source ~/.bashrc <br>
cd ~/DRL-robot-navigation/catkin_ws <br>
source devel_isolated/setup.bash <br>


## Run Code
cd ~/DRL-robot-navigation/TD3 <br>
python3 train_velodyne_td3.py

## Tensorboard
cd ~/DRL-robot-navigation/TD3 <br>
tensorboard --logdir 'runs'



## Run Gazebo Client to see the simulation
source /usr/share/gazebo/setup.sh <br>
gzclient <br>
% (or Check roslaunch setting to see Gazebo simulation client)

## Check Gazebo Run
gazebo --verbose

## Kill Gazebo
killall gzserver <br>
killall rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
