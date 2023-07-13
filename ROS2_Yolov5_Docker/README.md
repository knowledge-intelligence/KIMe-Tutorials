# ROS2 Yolov5 Docker Tutorial (in VirtualBox)
- Network connected.
- USB camera connected to VM.
- HDD >= 50GB for Docker build


## (Install ROS2 Foxy)
$ wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_ros2_foxy.sh && chmod 755 ./install_ros2_foxy.sh && ./install_ros2_foxy.sh


## (Install Docker)
$ wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_docker.sh && chmod 755 ./install_docker.sh && ./install_docker.sh


## (git clone)
$ git clone https://github.com/knowledge-intelligence/KIMe-Tutorials.git -b main


## (Modify ROS Domain ID)
$ export ROS_DOMAIN_ID=??? (0 ~ 101)
  
## (Check Camera in VirtualBox by Cheese app)
Using Ubuntu Software > Cheese App Install

## (Check Webcam Index)
(If needed) $ sudo apt install v4l-utils <br>
$ v4l2-ctl --list-devices

## (Modify the camera index in webcam_pub.py code)
```python
{
(Line 38 in webcam_pub.py) self.cap = cv2.VideoCapture(0)
}
```	

## (Build Dockerfile)
(GPU) $ sudo docker build . -t yolov5 <br>
(CPU) $ sudo docker build -f Dockerfile_CPU . -t yolov5_cpu


## (Run Docker Image)
(GPU) $ sudo docker run -it yolov5 --name yolov5_docker <br>
(CPU) $ sudo docker run -it yolov5_cpu --name yolov5_docker <br>

### Run Docker Image w/ ROS_DOMAIN_ID
$ sudo docker run -it --entrypoint /bin/bash yolov5_cpu -c "source /opt/ros/foxy/setup.bash && source ./install/setup.bash && export ROS_DOMAIN_ID=2 && ros2 run ros2_yolov5_docker ros2_yolov5_docker_node" --name yolov5_docker
<br>[참고] https://www.daleseo.com/docker-run/



## (Build ROS2 nodes for Out_Docker)
1. Go to "KIMe-Tutorials" folder
$ cd ~/KIMe-Tutorials/ROS2_Yolov5_Docker/Out_Docker/ <br>
2. Build
$ colcon build --symlink-install


## (Run Out_Docker Nodes - Publisher)
$ source ~/KIMe-Tutorials/ROS2_Yolov5_Docker/Out_Docker/install/setup.bash <br>
$ ros2 run ros2_yolov5 img_publisher


## (Run Out_Docker Nodes - Subscriber)
$ source ~/KIMe-Tutorials/ROS2_Yolov5_Docker/Out_Docker/install/setup.bash <br>
$ ros2 run ros2_yolov5 img_subscriber



# ETC

## (Docker Images/ps list)
$ sudo docker images list <br>
$ sudo docker ps <br>

## (To run a disposable new container / run it without --rm for no disposable)
(--rm 옵션, 일회성으로 실행) <br>
$ docker run --rm -it --entrypoint bash \<image-name-or-id\>

## (to enter a running container)
(-it 컨테이너를 종료하지 않고, 터미널의 입력을 계속해서 컨테이너로 전달하기 위해서 사용) <br>
$ docker exec -it \<container-name-or-id\> bash


## docker build no space left on device
(Ref) https://www.baeldung.com/linux/docker-fix-no-space-error
<br><br>
### 1. Finding the Current Storage Location
$ sudo docker info -f '{{ .DockerRootDir }}' <br>
/var/lib/docker <br>
<br>
### 2. Changing the Storage Location
$ mkdir -p /tmp/new-docker-root-dir <br>
$ sudo nano /etc/docker/daemon.json <br>
{ <br>
   "data-root": "/tmp/new-docker-root-dir" <br>
} <br>
<br>
### 3. Restarting & Confirming the Storage Location
$ sudo systemctl restart docker <br>
$ sudo docker info -f '{{ .DockerRootDir}}' <br>
/tmp/new-docker-root-dir <br>