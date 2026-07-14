# ROS2 Yolov5 Docker Tutorial (Jazzy Tested)

## Install cyclonedds
```bash
sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp
```

## Modify ROS Domain ID
```bash
echo $ROS_DOMAIN_ID
# or
export ROS_DOMAIN_ID=1 #(0 ~ 101)
```


## Check Webcam Index
### (If needed)
```bash
sudo apt install v4l-utils
```
```bash
v4l2-ctl --list-devices
```


## Build Dockerfile
- GPU
```bash
cd ~/yolo_ws/src
sudo docker build -f ./Dockerfile_opt . -t yolov5
```
- CPU
```bash
cd ~/yolo_ws/src
sudo docker build -f ./Dockerfile_CPU_opt . -t yolov5_cpu
```


## Run Docker Image - 호스트에서 먼저 X11 접근 허용 (컨테이너 root가 디스플레이에 그리도록)
```bash
xhost +local:docker

sudo docker run --rm -it \
  --net=host \
  --ipc=host \
  --pid=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=1 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/runtime-user:/tmp/runtime-user \
  --name yolov5-docker \
  yolov5


sudo docker run --rm -it \
  --net=host \
  --ipc=host \
  --pid=host \
  --gpus all \
  -e ROS_DOMAIN_ID=1 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  --name yolov5-docker \
  yolov5 \
  bash
```


## Install NVIDIA Container Toolkit
### 1. NVIDIA Container Toolkit 설치 (없는 경우)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### 2. Docker 런타임에 등록
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### 3. Docker 재시작
```bash
sudo systemctl restart docker
```




## Build ROS2 nodes for Out_Docker
```bash
source /opt/ros/jazzy/setup.bash
cd ~/yolo_ws/src
colcon build --symlink-install --packages-select ros2_yolov5
```

## Run Out_Docker Nodes - Publisher
```bash
source /opt/ros/jazzy/setup.bash
source ~/yolo_ws/src/install/setup.bash
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run ros2_yolov5 img_publisher --ros-args -p source:="/home/Administrator/yolo_ws/assets/TestVideo1.mp4"
```

```bash
# 기본 실행 (0번 카메라): 
ros2 run ros2_yolov5 img_publisher

# 특정 인덱스 카메라: 
ros2 run ros2_yolov5 img_publisher --ros-args -p source:=1

# 동영상 파일: 
ros2 run ros2_yolov5 img_publisher --ros-args -p source:="/path/to/video.mp4"
```

## Run Out_Docker Nodes - Subscriber
```bash
source /opt/ros/jazzy/setup.bash
source ~/yolo_ws/src/install/setup.bash
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 run ros2_yolov5 img_subscriber
```



# Docker 관련 CMD

## Docker Images/ps list
```bash
$ sudo docker images list
$ sudo docker ps
```

## GPU 접근이 실제로 되는지 확인
```bash
docker exec -it yolov5-docker nvidia-smi
```

## rm 옵션, 일회성으로 실행
```bash
docker run --rm -it --entrypoint bash \<image-name-or-id\>
docker run --rm -it --entrypoint bash yolov5


source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## to enter a running container
(-it 컨테이너를 종료하지 않고, 터미널의 입력을 계속해서 컨테이너로 전달하기 위해서 사용)
```bash
docker exec -it \<container-name-or-id\> bash
```