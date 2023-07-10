#!/bin/bash
# Gen @ 200829
# : Install Conda in Ubuntu 20.04

set -x

echo "[Install Conda]"
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6


curl -O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sha256sum Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
source ~/.bashrc
conda list

echo "[Complete!!!]"
exec bash
exit 0
