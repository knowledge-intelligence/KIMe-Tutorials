# yolov5_test
Yolo-v5 Test Repository


# 환경 설치
## Conda 가상환경 생성 (yolov5_test)
conda create -n yolov5_test python=3.9

## PyTorch 설치
### CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
### CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
### CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch


## Yolo v5 관련 설치
pip install -U ultralytics

## yolov5_test 실행

## PIL 오류시 (DLL load failed while importing _imaging: 지정된 모듈을 찾을 수 없습니다)
pip uninstall pillow <br>
pip install pillow==9.1.0

## Yolo v5 Github 복사
git clone https://github.com/ultralytics/yolov5.git

## Webcam Real-time Yolo v5 테스트
python yolov5/detect.py --weights yolov5s.pt --source 0
