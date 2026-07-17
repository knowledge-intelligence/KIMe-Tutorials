# KIMe-Tutorials
KIMe Tutorials (Isaac Sim/Lab, ROS2, AI)


## Install ROS2 Humble
```bash
wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/main/scripts/install_ros2_humble.sh && chmod 755 ./install_ros2_humble.sh && ./install_ros2_humble.sh
```

## Install Docker
```bash
wget https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/main/scripts/install_docker.sh && chmod 755 ./install_docker.sh && ./install_docker.sh
```

## Docker w/o sudo
```bash
sudo usermod -aG docker $USER && newgrp docker
```

## Docker Build
```bash
sudo docker build . –t yolov5
```

## Docker Run
```bash
sudo docker run –t yolov5
```

## remove all containers
```bash
sudo docker rm $(sudo docker ps -a -q)
```

## run a disposable new container
```bash
(To run a disposable new container / run it without --rm for no disposable)
(--rm 옵션, 일회성으로 실행)
docker run --rm -it --entrypoint bash <image-name-or-id>
```

## to enter a running container
```bash
(-it 컨테이너를 종료하지 않고, 터미널의 입력을 계속해서 컨테이너로 전달하기 위해서 사용)
docker exec -it <container-name-or-id> bash
```
