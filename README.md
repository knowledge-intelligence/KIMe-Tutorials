# KIMe-Tutorials for Yolov5 + ROS2 Tutorials

[Notion Link](https://kimelab.notion.site/KIMe-Lab-145875de525f80e98d63ff6e6637d037)

## Install Cheese for Camera Viewer & VSCode
```bash
sudo snap install cheese
sudo snap install --classic code
```

## Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```



## Install NVIDIA Container Toolkit
### 1. NVIDIA Container Toolkit 설치 (없는 경우)
```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### 2. Docker 런타임에 등록
```shell
sudo nvidia-ctk runtime configure --runtime=docker
```

### 3. Docker 재시작
```shell
sudo systemctl restart docker
```



## venv Set-Up
```bash
git clone -b yolov5-ros2 https://github.com/knowledge-intelligence/KIMe-Tutorials.git yolo_ws
cd yolo_ws
uv init
uv venv --python 3.10
```

## Install PyTorch & Yolov5
```bash
uv add torch --index-url https://download.pytorch.org/whl/cu124
uv add numpy pandas ultralytics tqdm seaborn scipy
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Test Run
```bash
uv run python src/yolov5_test.py
```

## Change Docker Data Location
```bash
chmod +x docker_relocate.sh
sudo ./docker_relocate.sh /Temp/data/docker
```


## 테스트 동영상
[pexels](https://www.pexels.com/ko-kr/)
[pixabay](https://pixabay.com/)


