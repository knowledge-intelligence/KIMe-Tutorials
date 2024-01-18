# KIMe-Tutorials
ROS1- &amp; ROS2- &amp; etc. Tutorials


(Install ROS1 Noetic)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_ros_noetic.sh && chmod 755 ./install_ros_noetic.sh && ./install_ros_noetic.sh


(Install ROS2 Foxy)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_ros2_foxy.sh && chmod 755 ./install_ros2_foxy.sh && ./install_ros2_foxy.sh

(Install ROS2 Humble)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_ros2_humble.sh && chmod 755 ./install_ros2_humble.sh && ./install_ros2_humble.sh


(Install Docker)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/master/install_docker.sh && chmod 755 ./install_docker.sh && ./install_docker.sh



(git clone)
$ git clone https://github.com/ros/ros_tutorials.git -b foxy-devel

(Docker Build)
$ sudo docker build . –t yolov5

(Docker Run)
$ sudo docker run –t yolov5

(remove all containers)
$ sudo docker rm $(sudo docker ps -a -q)

(To run a disposable new container / run it without --rm for no disposable)
(--rm 옵션, 일회성으로 실행)
$ docker run --rm -it --entrypoint bash <image-name-or-id>


(to enter a running container)
(-it 컨테이너를 종료하지 않고, 터미널의 입력을 계속해서 컨테이너로 전달하기 위해서 사용)
$ docker exec -it <container-name-or-id> bash

(Run Out_Docker Nodes)
$ source ~/robot_ws/install/setup.bash
$ ros2 run ros2_yolov5 img_publisher
$ ros2 run ros2_yolov5 img_subscriber


