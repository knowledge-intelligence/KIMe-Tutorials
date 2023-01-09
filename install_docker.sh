#!/bin/bash
# Gen @ 230109
# : Install Docker

set -x


# Ref: https://docs.docker.com/engine/install/ubuntu/

#Uninstall old versions
echo "[Uninstall old versions]"
sudo apt remove docker docker-engine docker.io containerd runc

#Set up the repository
echo "[Set up the repository]"
sudo apt update
sudo apt install -y \
  ca-certificates \
  curl \
  gnupg \
  lsb-release 

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

sh -c "echo \
  \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null"


sudo chmod a+r /etc/apt/keyrings/docker.gpg
sudo apt-get update


# Install Docker
echo "[Install Docker]"
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Hello Docker
echo "[Hello Docker]"
sudo docker run hello-world


# Done
echo "[Complete!!!]"
exec bash
exit 0
