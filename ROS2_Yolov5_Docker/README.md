# ROS2 Yolov5 Docker Tutorial (Humble Tested)

## Verify GPU Availability
```shell
python3
```
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
```

## Git Clone
```shell
mkdir -p ~/yolo_ws && cd ~/yolo_ws
git clone https://github.com/knowledge-intelligence/KIMe-Tutorials.git -b main
```

## Modify ROS Domain ID
```shell
echo $ROS_DOMAIN_ID
# or
export ROS_DOMAIN_ID=1 #(0 ~ 101)
```

## Check Camera by Cheese app
Using Ubuntu Software > Cheese App Install

## Check Webcam Index
### (If needed)
```shell
sudo apt install v4l-utils
```
```shell
v4l2-ctl --list-devices
```
## Modify the Webcam index in webcam_pub.py
```python
#(Line 38 in webcam_pub.py)
self.cap = cv2.VideoCapture(0)
```

## Build Dockerfile
```shell
cd ~/yolo_ws/KIMe-Tutorials/ROS2_Yolov5_Docker
sudo docker build -f Dockerfile . -t yolov5
```

## Run Docker Image
```shell
sudo docker run --rm -it --privileged \
   --net=host -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix  \
   -v /tmp/runtime-user:/tmp/runtime-user \
   --name yolov5-docker yolov5
```

## Build ROS2 nodes for Out_Docker
```shell
cd ~/yolo_ws/KIMe-Tutorials/ROS2_Yolov5_Docker
colcon build --symlink-install --packages-select ros2_yolov5
```

## Run Out_Docker Nodes - Publisher
```shell
source ~/yolo_ws/KIMe-Tutorials/ROS2_Yolov5_Docker/install/setup.bash
ros2 run ros2_yolov5 img_publisher
```

## Run Out_Docker Nodes - Subscriber
```shell
source ~/yolo_ws/KIMe-Tutorials/ROS2_Yolov5_Docker/install/setup.bash
ros2 run ros2_yolov5 img_subscriber
```



# Ref
- https://duvallee.tistory.com/26
- https://answers.ros.org/question/416582/



# ETC

## (Docker Images/ps list)
```shell
$ sudo docker images list
$ sudo docker ps
```
## (To run a disposable new container / run it without --rm for no disposable)

(--rm 옵션, 일회성으로 실행) <br>
```shell
$ docker run --rm -it --entrypoint bash \<image-name-or-id\>
```

## (to enter a running container)
(-it 컨테이너를 종료하지 않고, 터미널의 입력을 계속해서 컨테이너로 전달하기 위해서 사용) <br>
```shell
docker exec -it \<container-name-or-id\> bash
```

## docker build no space left on device
(Ref) https://www.baeldung.com/linux/docker-fix-no-space-error
<br><br>
### 1. Finding the Current Storage Location
```shell
sudo docker info -f '{{ .DockerRootDir }}' <br>
```
> /var/lib/docker <br>

### 2. Changing the Storage Location
```shell
mkdir -p /tmp/new-docker-root-dir
sudo nano /etc/docker/daemon.json
```
```
{
   "data-root": "/tmp/new-docker-root-dir"
}
```
### 3. Restarting & Confirming the Storage Location
```shell
sudo systemctl restart docker
sudo docker info -f '{{ .DockerRootDir}}'
```
> /tmp/new-docker-root-dir
