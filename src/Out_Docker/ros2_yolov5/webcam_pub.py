# 웹캠 또는 동영상 파일로부터 실시간 스트리밍 영상을
# 퍼블리시하는 기본 ROS 2 프로그램


# 필요한 라이브러리 임포트
import rclpy # ROS 2용 파이썬 클라이언트 라이브러리
from rclpy.node import Node # 노드 생성을 담당
from sensor_msgs.msg import Image # Image는 메시지 타입
from cv_bridge import CvBridge # ROS와 OpenCV 이미지 간 변환 패키지
import cv2 # OpenCV 라이브러리


class ImagePublisher(Node):
  """
  Node 클래스의 서브클래스인 ImagePublisher 클래스를 생성합니다.
  """
  def __init__(self):
    """
    노드를 설정하기 위한 클래스 생성자
    """
    # Node 클래스의 생성자를 호출하고 이름을 부여
    super().__init__('image_publisher')

    # 퍼블리셔 생성. 이 퍼블리셔는 yolov5/image_raw 토픽으로
    # Image를 퍼블리시합니다. 큐 크기는 10개 메시지입니다.
    self.publisher_ = self.create_publisher(Image, 'yolov5/image_raw', 10)

    # 소스 파라미터 선언
    # - 숫자 문자열(예: '0', '1')이면 카메라 인덱스로 사용
    # - 그 외 문자열이면 동영상 파일 경로로 사용
    self.declare_parameter('source', '0')
    source_param = self.get_parameter('source').get_parameter_value().string_value

    self.is_video_file = not source_param.isdigit()
    if self.is_video_file:
      source = source_param
    else:
      source = int(source_param)

    # VideoCapture 객체 생성
    self.cap = cv2.VideoCapture(source)

    if not self.cap.isOpened():
      self.get_logger().error(f'Failed to open source: {source_param}')

    # 동영상 파일인 경우 원본 FPS에 맞춰 퍼블리시 주기를 설정하고,
    # 그 외(카메라)에는 기본 0.1초 주기를 사용합니다.
    timer_period = 0.1  # 초 단위
    if self.is_video_file:
      fps = self.cap.get(cv2.CAP_PROP_FPS)
      if fps and fps > 0:
        timer_period = 1.0 / fps

    # 타이머 생성
    self.timer = self.create_timer(timer_period, self.timer_callback)

    # ROS와 OpenCV 이미지 간 변환에 사용됩니다
    self.br = CvBridge()

  def timer_callback(self):
    """
    콜백 함수.
    타이머 주기마다 호출됩니다.
    """
    # 프레임을 한 장씩 캡처
    # 이 메서드는 비디오 프레임과 함께
    # True/False 값도 반환합니다.
    ret, frame = self.cap.read()

    # 동영상 파일의 끝에 도달하면 처음으로 되감아 계속 재생합니다.
    if not ret and self.is_video_file:
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      ret, frame = self.cap.read()

    if ret == True:
      # 프레임을 가로 640px 기준으로 축소합니다.
      # 원본 종횡비를 유지하도록 세로 크기도 비례해서 계산합니다.
      target_width = 640
      h, w = frame.shape[:2]
      if w > target_width:
        target_height = int(h * target_width / w)
        frame = cv2.resize(frame, (target_width, target_height),
                           interpolation=cv2.INTER_AREA)

      # 이미지를 퍼블리시합니다.
      # 'cv2_to_imgmsg' 메서드는 OpenCV
      # 이미지를 ROS 2 이미지 메시지로 변환합니다
      self.publisher_.publish(self.br.cv2_to_imgmsg(frame))

      # 콘솔에 메시지 출력
      self.get_logger().info('Publishing video frame')

    else:
      # 콘솔에 에러 메시지 출력
      self.get_logger().info('Capturing failed')



def main(args=None):

  # rclpy 라이브러리 초기화
  rclpy.init(args=args)

  # 노드 생성
  image_publisher = ImagePublisher()

  # 콜백 함수가 호출되도록 노드를 스핀
  rclpy.spin(image_publisher)

  # 노드를 명시적으로 파괴
  # (선택 사항 - 하지 않아도 가비지 컬렉터가
  # 노드 객체를 파괴할 때 자동으로 처리됩니다)
  image_publisher.destroy_node()

  # 파이썬용 ROS 클라이언트 라이브러리 종료
  rclpy.shutdown()

if __name__ == '__main__':
  main()
