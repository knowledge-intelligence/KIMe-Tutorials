# KIMe-Tutorials for Yolov5 + ROS2 Tutorials

## Install Cheese for Camera Viewer
```bash
sudo snap install cheese
```

## Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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
uv run python yolov5_test.py
```

## 테스트 동영상
[pexels](https://www.pexels.com/ko-kr/)
[pixabay](https://pixabay.com/)
