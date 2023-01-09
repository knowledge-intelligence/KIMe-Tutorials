from setuptools import setup

package_name = 'ros2_yolov5'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kime',
    maintainer_email='stshin@dau.ac.kr',
    description='ROS2_Yolov5',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
          'img_publisher = ros2_yolov5.webcam_pub:main',
          'img_subscriber = ros2_yolov5.webcam_sub:main',
        ],
    },
)
