# ROS2 Yolov5 Docker Tutorial (in VirtualBox)
- Network connected.
- USB camera connected to VM.
- HDD >= 50GB for Docker build


## (Install ROS2 Foxy)
```shell
wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_ros2_foxy.sh && chmod 755 ./install_ros2_foxy.sh && ./install_ros2_foxy.sh
```

## (Install Docker)
```shell
wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_docker.sh && chmod 755 ./install_docker.sh && ./install_docker.sh
```

## (git clone)
```shell
git clone https://github.com/knowledge-intelligence/KIMe-Tutorials.git -b main
```

## (Modify ROS Domain ID)
```shell
export ROS_DOMAIN_ID=??? (0 ~ 101)
```
## (Check Camera in VirtualBox by Cheese app)
Using Ubuntu Software > Cheese App Install

## (Check Webcam Index)
```shell
(If needed) sudo apt install v4l-utils
```
```shell
v4l2-ctl --list-devices
```
## (Modify the camera index in webcam_pub.py code)
```python
(Line 38 in webcam_pub.py) 
self.cap = cv2.VideoCapture(0)
```	

## (Build Dockerfile)
```shell
(GPU) $ sudo docker build . -t yolov5
(CPU) $ sudo docker build -f Dockerfile_CPU . -t yolov5_cpu
```

## (Run Docker Image)
```shell
(GPU) $ sudo docker run -it yolov5 --name yolov5_docker
(CPU) $ sudo docker run -it yolov5_cpu --name yolov5_docker
```
### Run Docker Image w/ ROS_DOMAIN_ID
```shell
sudo docker run -it --entrypoint /bin/bash yolov5_cpu -c "source /opt/ros/foxy/setup.bash && source ./install/setup.bash && export ROS_DOMAIN_ID=2 && ros2 run ros2_yolov5_docker ros2_yolov5_docker_node" --name yolov5_docker
```
[참고] https://www.daleseo.com/docker-run/



## (Build ROS2 nodes for Out_Docker)
1. Go to "KIMe-Tutorials" folder <br>
$ cd ~/KIMe-Tutorials/ROS2_Yolov5_Docker/Out_Docker/ <br>
2. Build <br>
$ colcon build --symlink-install


## (Run Out_Docker Nodes - Publisher)
```shell
$ source ~/KIMe-Tutorials/ROS2_Yolov5_Docker/Out_Docker/install/setup.bash
$ ros2 run ros2_yolov5 img_publisher
```

## (Run Out_Docker Nodes - Subscriber)
```shell
$ source ~/KIMe-Tutorials/ROS2_Yolov5_Docker/Out_Docker/install/setup.bash
$ ros2 run ros2_yolov5 img_subscriber
```

<br><br><br><br>
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
