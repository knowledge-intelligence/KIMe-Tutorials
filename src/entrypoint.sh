#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /app/kime_ws/install/setup.bash
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
exec "$@"