# ROS2 패키지 기초 (Python - Topic)

## Linux CMD 파일 메니져
```shell
$ mc
$ mcedit
```

## 1. 패키지 생성
```shell
$ mkdir -p ~/robot_ws/src
$ cd ~/robot_ws/src
$ ros2 pkg create my_first_ros_rclpy_pkg --build-type ament_python --dependencies rclpy std_msgs
```

## 2. 패키지 설정
```shell
$ cd ~/robot_ws/src/my_first_ros_rclpy_pkg
$ mcedit setup.py
```

```python
from setuptools import find_packages
from setuptools import setup

package_name = 'my_first_ros_rclpy_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='KIMe’,
    author_email='KIMe@KIMe.com',
    maintainer='KIMe',
    maintainer_email='KIMe@KIMe.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
   description='ROS2 rclpy basic package',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'helloworld_publisher = my_first_ros_rclpy_pkg.helloworld_publisher:main',
            'helloworld_subscriber = my_first_ros_rclpy_pkg.helloworld_subscriber:main',
        ],
    })
```

## 3. 퍼블리셔 노드 작성
```shell
$ cd ~/robot_ws/src/my_first_ros_rclpy_pkg/my_first_ros_rclpy_pkg
$ mcedit helloworld_publisher.py
```

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

class HelloworldPublisher(Node):
    def __init__(self):
        super().__init__('helloworld_publisher')
        qos_profile = QoSProfile(depth=10)
        self.helloworld_publisher = self.create_publisher(String, 'helloworld', qos_profile)
        self.timer = self.create_timer(1, self.publish_helloworld_msg)
        self.count = 0

    def publish_helloworld_msg(self):
        msg = String()
        msg.data = 'Hello World: {0}'.format(self.count)
        self.helloworld_publisher.publish(msg)
        self.get_logger().info('Published message: {0}'.format(msg.data))
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = HelloworldPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. 서브스크라이버 노드 작성
```shell
$ cd ~/robot_ws/src/my_first_ros_rclpy_pkg/my_first_ros_rclpy_pkg
$ mcedit helloworld_subscriber.py
```

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class HelloworldSubscriber(Node):

    def __init__(self):
        super().__init__('Helloworld_subscriber')
        qos_profile = QoSProfile(depth=10)
        self.helloworld_subscriber = self.create_subscription(
            String,
            'helloworld',
            self.subscribe_topic_message,
            qos_profile)

    def subscribe_topic_message(self, msg):
        self.get_logger().info('Received message: {0}'.format(msg.data))

def main(args=None):
    rclpy.init(args=args)
    node = HelloworldSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. 빌드
```shell
$ cd ~/robot_ws && colcon build --symlink-install --packages-select my_first_ros_rclpy_pkg
```

## 6. 실행
```shell
$ source ~/robot_ws/install/local_setup.bash
$ ros2 run my_first_ros_rclpy_pkg helloworld_subscriber
```
```shell
$ source ~/robot_ws/install/local_setup.bash
$ ros2 run my_first_ros_rclpy_pkg helloworld_publisher
```