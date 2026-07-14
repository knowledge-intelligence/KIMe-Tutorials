# 내장 웹캠으로부터 실시간 스트리밍 영상을
# 구독하는 기본 ROS 2 프로그램


# 필요한 라이브러리 임포트
import rclpy # ROS 2용 파이썬 라이브러리
from rclpy.node import Node # 노드 생성을 담당
from sensor_msgs.msg import Image # Image는 메시지 타입
from cv_bridge import CvBridge # ROS와 OpenCV 이미지 간 변환 패키지
import cv2 # OpenCV 라이브러리

class ImageSubscriber(Node):
  """
  Node 클래스의 서브클래스인 ImageSubscriber 클래스를 생성합니다.
  """
  def __init__(self):
    """
    노드를 설정하기 위한 클래스 생성자
    """
    # Node 클래스의 생성자를 호출하고 이름을 부여
    super().__init__('image_subscriber')

    # 구독자 생성. 이 구독자는 video_frames 토픽으로부터
    # Image를 수신합니다. 큐 크기는 10개 메시지입니다.
    self.subscription = self.create_subscription(
      Image,
      'yolov5/image',
      self.listener_callback,
      10)
    self.subscription # 사용하지 않는 변수 경고 방지

    # ROS와 OpenCV 이미지 간 변환에 사용됩니다
    self.br = CvBridge()

  def listener_callback(self, data):
    """
    콜백 함수.
    """
    # 콘솔에 메시지 출력
    self.get_logger().info('Receiving video frame')

    # ROS Image 메시지를 OpenCV 이미지로 변환
    current_frame = self.br.imgmsg_to_cv2(data)

    # 이미지 표시
    cv2.imshow("yolov5 processed camera", current_frame)
    cv2.waitKey(1)

def main(args=None):

  # rclpy 라이브러리 초기화
  rclpy.init(args=args)

  # 노드 생성
  image_subscriber = ImageSubscriber()

  # 콜백 함수가 호출되도록 노드를 스핀
  rclpy.spin(image_subscriber)

  # 노드를 명시적으로 파괴
  # (선택 사항 - 하지 않아도 가비지 컬렉터가
  # 노드 객체를 파괴할 때 자동으로 처리됩니다)
  image_subscriber.destroy_node()

  # 파이썬용 ROS 클라이언트 라이브러리 종료
  rclpy.shutdown()

if __name__ == '__main__':
  main()
