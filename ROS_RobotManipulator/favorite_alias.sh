#!/bin/bash
# @ 220819
# : Favorite Alias

set -x

echo "[Set Alias to Bashrc]"

sh -c "echo \"export CONDA_AUTO_ACTIVATE_BASE=false\" >> ~/.bashrc"
sh -c "echo \"alias dofoxy='source /opt/ros/foxy/setup.bash; source ~/robot_ws/install/local_setup.bash; echo \"Activate foxy!\"'\" >> ~/.bashrc"
sh -c "echo \"alias donoetic='source /opt/ros/noetic/setup.bash; source ~/ros1_ws/devel/setup.bash; echo \"Activate noetic!\"'\" >> ~/.bashrc"
sh -c "echo \"alias doconda='conda activate test'\" >> ~/.bashrc"


echo "[Complete!!!]"
exec bash
exit 0
