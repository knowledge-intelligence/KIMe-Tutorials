# ROS2 패키지 설계 (Python - Topic/Service/Action/Parameter)

## GitHub 주소
https://github.com/robotpilot/ros2-seminar-examples

## 1. 소스 코드 다운로드 및 빌드
```bash
mkdir -p ~/robot_ws/src
cd ~/robot_ws/src
git clone https://github.com/robotpilot/ros2-seminar-examples.git
cd ~/robot_ws && colcon build --symlink-install --packages-up-to topic_service_action_rclpy_example
```
```bash
# 특정 폴더 제외 삭제
rm -rf !(msg_srv_action_interface_example|topic_service_action_rclpy_example)
```

## 2. 실행
```bash
source ~/robot_ws/install/setup.bash
ros2 run topic_service_action_rclpy_example calculator
```
```bash
source ~/robot_ws/install/setup.bash
ros2 run topic_service_action_rclpy_example argument
```
```bash
source ~/robot_ws/install/setup.bash
ros2 run topic_service_action_rclpy_example operator
```
```bash
source ~/robot_ws/install/setup.bash
ros2 run topic_service_action_rclpy_example checker
ros2 run topic_service_action_rclpy_example checker -g 100
```

## 6. 런치 파일 실행
```bash
ros2 launch topic_service_action_rclpy_example arithmetic.launch.py
```
