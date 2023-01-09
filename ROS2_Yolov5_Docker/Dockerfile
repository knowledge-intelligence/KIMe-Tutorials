FROM shaderobotics/ros:foxy-cudnn

# create a basic workspace
RUN ["/bin/bash", "-c", "mkdir -p /app/shade_ws/src"]
WORKDIR /app/kime_ws

# install some basic tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*
       
# Install basic messaging and image processing tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-pip \
    ros-foxy-cv-bridge \
    ros-foxy-vision-msgs \
    python3-natsort \
    ros-foxy-vision-opencv && \
    rm -rf /var/lib/apt/lists/*
   
# Utilize pip to install any required python dependencies
RUN pip3 install --upgrade pip
#RUN python3 -m pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
RUN pip3 install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

COPY ./In_Docker /app/kime_ws/ROS2_Yolov5_Docker
COPY ./In_Docker/yolov5s.pt /app/kime_ws/
RUN colcon build

# begin the node
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/foxy/setup.bash && source ./install/setup.bash && ros2 run ros2_yolov5_docker ros2_yolov5_docker_node"]
