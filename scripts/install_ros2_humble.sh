#!/bin/bash
# Gen @ 200829
# Upd @ 260708
# : Install ROS2 Humble Hawksbill in Ubuntu 22.04

set -x

# Without this, tzdata (pulled in as a dependency partway through the ros-humble-desktop
# install) can hit a live debconf "Geographic area:" prompt and hang the script forever.
# `export DEBIAN_FRONTEND` alone is not enough since sudo resets the environment by
# default, so pin debconf's own frontend selection instead - that persists regardless
# of how apt is invoked.
echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
export DEBIAN_FRONTEND=noninteractive

name_ws="robot_ws"
name_ros2_distro="humble"


echo "[Setup Locales]"
sudo apt update && sudo apt install -y locales

sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "[Setup Sources]"
# Ref: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe

sudo apt update && sudo apt install -y curl
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

echo "[Installing ROS2]"
sudo apt update
# NOTE: on a fresh Ubuntu 22.04 install, systemd/udev must be up to date before installing ROS2,
# otherwise apt may remove critical system packages. Ref: https://github.com/ros2/ros2/issues/1272
sudo apt upgrade -y
sudo apt install -y ros-$name_ros2_distro-desktop ros-$name_ros2_distro-rmw-fastrtps* ros-$name_ros2_distro-rmw-cyclonedds*

echo "[Installing ROS2 Tools]"
sudo apt update && sudo apt install -y \
  ros-dev-tools \
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
add_bashrc_line() {
  grep -qxF "$1" ~/.bashrc || echo "$1" >> ~/.bashrc
}

add_bashrc_line "alias do${name_ros2_distro}='source /opt/ros/${name_ros2_distro}/setup.bash; echo Activate ${name_ros2_distro}!'"

# add_bashrc_line "source /opt/ros/${name_ros2_distro}/setup.bash"
# add_bashrc_line "source ~/${name_ws}/install/local_setup.bash"
# add_bashrc_line "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash"
# add_bashrc_line "source /usr/share/vcstool-completion/vcs.bash"
# add_bashrc_line "source /usr/share/colcon_cd/function/colcon_cd.sh"

add_bashrc_line "export _colcon_cd_root=~/${name_ws}"

add_bashrc_line "export ROS_DOMAIN_ID=0"
# add_bashrc_line "export ROS_NAMESPACE=robot1"

add_bashrc_line "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export RMW_IMPLEMENTATION=rmw_connext_cpp
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# export RMW_IMPLEMENTATION=rmw_gurumdds_cpp

# export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity} {time}] [{name}]: {message} ({function_name}() at {file_name}:{line_number})'
# add_bashrc_line "export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}]: {message}'"
# add_bashrc_line "export RCUTILS_COLORIZED_OUTPUT=1"
# add_bashrc_line "export RCUTILS_LOGGING_USE_STDOUT=0"
# add_bashrc_line "export RCUTILS_LOGGING_BUFFERED_STREAM=1"


echo "[Complete!!!]"
exec bash
