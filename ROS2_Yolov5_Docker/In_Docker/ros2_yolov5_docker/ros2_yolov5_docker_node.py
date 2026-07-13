import torch
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge  # ROS Image 메시지 <-> OpenCV(numpy) 이미지 변환
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2 # OpenCV library

class ImageSubscriber(Node):
    def __init__(self):
        # 노드 이름을 'yolov5_node'로 지정하여 초기화
        super().__init__('yolov5_node')

        # 'yolov5/image_raw' 토픽을 구독.
        # webcam_pub 노드가 퍼블리시하는 원본(미처리) 카메라 프레임을 수신한다.
        # 큐 사이즈 1: 처리 속도가 카메라 프레임레이트보다 느릴 수 있으므로
        # 오래된 프레임이 쌓이지 않도록 최신 프레임 1개만 유지한다.
        self.subscription = self.create_subscription(
            Image,
            'yolov5/image_raw',
            self.listener_callback,
            1)

        # YOLOv5 추론 결과(바운딩 박스가 그려진 이미지)를 퍼블리시할 퍼블리셔.
        # webcam_sub 노드가 'yolov5/image' 토픽을 구독해서 화면에 표시한다.
        self.image_publisher = self.create_publisher(Image, 'yolov5/image', 10)

        # ROS Image 메시지 <-> OpenCV 이미지 변환 유틸리티
        self.br = CvBridge()

        # torch.hub를 통해 ultralytics/yolov5 저장소에서 사전학습된 yolov5s 모델을 로드.
        # (컨테이너 빌드 시 함께 복사된 yolov5s.pt 가중치 파일을 활용)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.get_logger().info("Node Initialized")

    def listener_callback(self, data):
        # 구독한 프레임이 도착할 때마다 호출되는 콜백 함수
        self.get_logger().info("Got Image")

        # ROS Image 메시지를 OpenCV(numpy) 이미지로 변환
        current_frame = self.br.imgmsg_to_cv2(data)

        # GPU 연산 시 자동 혼합 정밀도(FP16/FP32 자동 전환)를 사용해
        # 추론 속도를 높이고 메모리 사용량을 줄인다. (CUDA 사용 가능 환경 전용)
        with torch.amp.autocast(device_type='cuda'):
            # 이미지를 모델에 통과시켜 객체 탐지 수행
            processed_image = self.model(current_frame)
        #results = self.br.cv2_to_imgmsg(processed_image.ims[0]) # Original Img
        # 탐지 결과를 원본 이미지 위에 바운딩 박스/라벨로 렌더링한 이미지를 ROS 메시지로 변환
        results = self.br.cv2_to_imgmsg(processed_image.render()[0]) # Boxed Img

        # 처리된(박스가 그려진) 이미지를 퍼블리시
        self.image_publisher.publish(results)

def main(args=None):
    # rclpy 라이브러리 초기화
    rclpy.init(args=args)

    # 노드 생성
    image_subscriber = ImageSubscriber()

    # 콜백 함수가 호출되도록 노드를 스핀 (구독 대기 상태 유지)
    rclpy.spin(image_subscriber)

    # 노드를 명시적으로 파괴
    image_subscriber.destroy_node()

    # 파이썬용 ROS 클라이언트 라이브러리 종료
    rclpy.shutdown()

if __name__ == '__main__':
    main()
