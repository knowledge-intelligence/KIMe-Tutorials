#!/bin/bash
# Gen @ 200829
# : Install ROS Noetic in Ubuntu 18.04 with VS Code

set -x

name_catkinws="catkin_ws"
name_ros_distro="melodic"
version=`lsb_release -sc`


echo "[Update & upgrade the package]"
sudo apt-get update
sudo apt-get upgrade

echo "[Installing chrony and setting the ntpdate]"
sudo apt-get install -y chrony ntpdate
sudo ntpdate ntp.ubuntu.com

echo "[Add the ROS repository]"
if [ ! -e /etc/apt/sources.list.d/ros-latest.list ]; then
  sudo sh -c "echo \"deb http://packages.ros.org/ros/ubuntu ${version} main\" > /etc/apt/sources.list.d/ros-latest.list"
fi

echo "[Download the ROS keys]"
roskey=`apt-key list | grep "ROS builder"`
if [ -z "$roskey" ]; then
  wget --quiet https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add -
fi

echo "[Update & upgrade the package]"
sudo apt-get update
sudo apt-get upgrade

echo "[Installing ROS]"
sudo apt-get install -y ros-$name_ros_distro-desktop-full ros-$name_ros_distro-rqt-*

echo "[rosdep init and python-rosinstall]"
sudo sh -c "rosdep init"
rosdep update
. /opt/ros/$name_ros_distro/setup.sh
sudo apt-get install -y python3-rosinstall
#sudo apt install python3-roslaunch
source /opt/ros/noetic/setup.bash

echo "[Making the catkin workspace and testing the catkin_make]"
mkdir -p ~/$name_catkinws/src
cd ~/$name_catkinws/src
catkin_init_workspace
cd ~/$name_catkinws/
catkin_make

echo "[Setting the ROS evironment]"
sh -c "echo \"source /opt/ros/$name_ros_distro/setup.bash\" >> ~/.bashrc"
sh -c "echo \"source ~/$name_catkinws/devel/setup.bash\" >> ~/.bashrc"
sh -c "echo \"export ROS_MASTER_URI=http://localhost:11311\" >> ~/.bashrc"
sh -c "echo \"export ROS_HOSTNAME=localhost\" >> ~/.bashrc"

echo "[Complete!!!]"

exec bash

exit 0

