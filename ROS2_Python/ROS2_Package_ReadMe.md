# ROS2 패키지 설계 (Python)



## GitHub 주소
https://github.com/robotpilot/ros2-seminar-examples

## Linux CMD 파일 메니져
$ mc

## 5. 소스 코드 다운로드 및 빌드
```shell
$ cd ~/robot_ws/src
$ git clone https://github.com/robotpilot/ros2-seminar-examples.git
$ cd ~/robot_ws && colcon build --symlink-install
```

## 6. 실행
```shell
$ ros2 run topic_service_action_rclpy_example calculator
$ ros2 run topic_service_action_rclpy_example argument
$ ros2 run topic_service_action_rclpy_example calculator
$ ros2 run topic_service_action_rclpy_example operator
$ ros2 run topic_service_action_rclpy_example checker
$ ros2 run topic_service_action_rclpy_example checker -g 100
```

## 6. 런치 파일 실행
```shell
$ ros2 launch topic_service_action_rclpy_example arithmetic.launch.py
```
