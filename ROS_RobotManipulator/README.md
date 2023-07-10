# ROS_RobotManipulator
Classes of Robot Engineering &amp; AI-based Unmanned Vehicle for Dong-A University <br>
동아대학교 기계공학과 로봇공학 수업 및 무인이동체 실습 프로그램 운영(인공지능 기반 로봇)(비교과)를 위한 코드


(예제)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/ROS_RobotManipulator/master/ros_noetic_install.sh && chmod 755 ./ros_noetic_install.sh && ./ros_noetic_install.sh

(예제)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/ROS_RobotManipulator/master/ros2_foxy_install.sh && chmod 755 ./ros2_foxy_install.sh && ./ros2_foxy_install.sh

(예제)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/ROS_RobotManipulator/master/vs_install.sh && chmod 755 ./vs_install.sh && ./vs_install.sh

(예제)<br>
$ wget https://raw.githubusercontent.com/knowledge-intelligence/ROS_RobotManipulator/master/conda_install.sh && chmod 755 ./conda_install.sh && ./conda_install.sh


(Conda Env Create)<br>
$ conda env export > env_requirements.txt
$ conda env create -f env_requirements.txt
$ conda list -e > conda_requirements.txt
$ conda create --name <environment_name> --file conda_requirements.txt
$ pip freeze > pip_requirements.txt
$ pip install -r pip_requirements.txt


(OneDrive in Ubuntu)<br>
https://itslinuxfoss.com/how-to-install-and-use-onedrive-on-ubuntu-20-04/<br>
[Install]<br>
$ wget -qO - https://download.opensuse.org/repositories/home:/npreining:/debian-ubuntu-onedrive/xUbuntu_20.04/Release.key | sudo apt-key add -
<br>
$ echo 'deb https://download.opensuse.org/repositories/home:/npreining:/debian-ubuntu-onedrive/xUbuntu_20.04/ ./' | sudo tee /etc/apt/sources.list.d/onedrive.list
<br>
$ sudo apt update
<br>
$ sudo apt install onedrive -y
<br>
<br>
[Run w/ Specific Folder]<br>
$ onedrive --synchronize --single-directory FOLDERNAME


(GPU Monitoring in Ubuntu)<br>
$ watch -d -n 0.5 nvidia-smi



(Conda + Pytorch 설치) <br>
[Install Pytorch + cudatoolkit] <br>
https://pytorch.org/get-started/locally/ <br>
$ conda config --set auto_activate_base False <br>
$ conda config --set channel_priority flexible (or strict)<br>
$ conda create -n test <br>
$ sudo apt update <br>
$ sudo apt upgrade <br>
$ conda install python=3.6
$ conda install seaborn scipy scikit-learn numpy matplotlib <br>
$ conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge <br>
$ conda install -c conda-forge dotmap <br>
$ conda install -c conda-forge ray-tune <br>
$ pip install tensorflow-gpu (w/ python3.10) <br>
$ conda install -c anaconda tensorflow-gpu (w/ python3.9)<br>
$ conda install -c conda-forge imbalanced-learn torchinfo
