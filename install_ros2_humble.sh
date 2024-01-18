#!/bin/bash
# Gen @ 200829
# : Install ROS2 Foxy Fitzroy in Ubuntu 20.04 with VS Code

set -x

name_ws="robot_ws"
name_ros2_distro="humble"
version=`lsb_release -sc`


echo "[Setup Locales]"
sudo apt update && sudo apt install -y locales

sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "[Setup Sources]"
sudo apt update && sudo apt install -y curl gnupg2 lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg


#echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${version} main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null


echo "[Installing ROS2]"
sudo apt update
sudo apt install -y ros-$name_ros2_distro-desktop ros-$name_ros2_distro-rmw-fastrtps* ros-$name_ros2_distro-rmw-cyclonedds*

# Delete Cache
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lcok*
sudo dpkg --configure -a
sudo apt update

echo "[Installing ROS2 Tools]"
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  libbullet-dev \
  python3-colcon-common-extensions \
  python3-flake8 \
  python3-pip \
  python3-pytest-cov \
  python3-rosdep \
  python3-setuptools \
  python3-vcstool \
  wget

python3 -m pip install -U \
  argcomplete \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest-repeat \
  pytest-rerunfailures \
  pytest

sudo apt install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev \
  libcunit1-dev



echo "[Making the catkin workspace and testing the catkin_make]"
source /opt/ros/$name_ros2_distro/setup.bash
mkdir -p ~/$name_ws/src
cd ~/$name_ws/
colcon build --symlink-install



echo "[Setting the ROS evironment]"
sh -c "echo \"alias dofoxy='source /opt/ros/${name_ros2_distro}/setup.bash; echo \"Activate foxy!\"'\" >> ~/.bashrc"

# sh -c "echo \"source /opt/ros/${name_ros2_distro}/setup.bash\" >> ~/.bashrc"
# sh -c "echo \"source ~/${name_ws}/install/local_setup.bash\" >> ~/.bashrc"
# sh -c "echo \"source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash\" >> ~/.bashrc"
# sh -c "echo \"source /usr/share/vcstool-completion/vcs.bash\" >> ~/.bashrc"
# sh -c "echo \"source /usr/share/colcon_cd/function/colcon_cd.sh\" >> ~/.bashrc"

sh -c "echo \"export _colcon_cd_root=~/${name_ws}\" >> ~/.bashrc"

sh -c "echo \"export ROS_DOMAIN_ID=0\" >> ~/.bashrc"
# sh -c "echo \"export ROS_NAMESPACE=robot1\" >> ~/.bashrc"

sh -c "echo \"export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\" >> ~/.bashrc"
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export RMW_IMPLEMENTATION=rmw_connext_cpp
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export RMW_IMPLEMENTATION=rmw_gurumdds_cpp

# export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity} {time}] [{name}]: {message} ({function_name}() at {file_name}:{line_number})'
# sh -c "echo \"export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}]: {message}'\" >> ~/.bashrc"
# sh -c "echo \"export RCUTILS_COLORIZED_OUTPUT=1\" >> ~/.bashrc"
# sh -c "echo \"export RCUTILS_LOGGING_USE_STDOUT=0\" >> ~/.bashrc"
# sh -c "echo \"export RCUTILS_LOGGING_BUFFERED_STREAM=1\" >> ~/.bashrc"


echo "[Complete!!!]"
exec bash
exit 0
