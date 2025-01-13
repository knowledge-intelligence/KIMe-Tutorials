#FROM ubuntu:22.04
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
FROM ros:humble

# ROS2 Desktop version
RUN apt-get update
RUN apt-get install -y \
    python3-pip
RUN pip install -U ultralytics
RUN pip install -U "numpy<2"

RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libbullet-dev \
    curl \
    ffmpeg \
    python3-natsort \
    python3-colcon-common-extensions \
    python3-flake8 \
    python3-pytest-cov \
    python3-rosdep \
    python3-setuptools \
    python3-vcstool \
    python3-colcon-common-extensions \
    python3-vcstool

RUN apt-get install -y \
    ros-humble-desktop \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-rmw-fastrtps* \
    ros-humble-rmw-cyclonedds* \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-vision-opencv


# create a basic workspace
RUN ["/bin/bash", "-c", "mkdir -p /app/kime_ws"]
WORKDIR /app/kime_ws


# # Utilize pip to install any required python dependencies
# RUN pip3 install --upgrade pip

# #RUN python3 -m pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
# RUN pip3 install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

COPY ./In_Docker /app/kime_ws/ROS2_Yolov5_Docker
COPY ./In_Docker/yolov5s.pt /app/kime_ws/
RUN colcon build

RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
RUN echo 'source ./install/setup.bash ' >> ~/.bashrc

ENV ROS_DOMAIN_ID=1
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV PYTHONWARNINGS="ignore::FutureWarning"

# begin the node
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source ./install/setup.bash && export ROS_DOMAIN_ID=1 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export PYTHONWARNINGS='ignore::FutureWarning' && ros2 run ros2_yolov5_docker ros2_yolov5_docker_node"]
#ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && ros2 run ros2_yolov5_docker ros2_yolov5_docker_node"]
# Run bash
CMD ["bash"]
