import torch
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2 # OpenCV library

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('yolov5_node')
        self.subscription = self.create_subscription(
            Image,
            'yolov5/image_raw',
            self.listener_callback,
            1)
        self.image_publisher = self.create_publisher(Image, 'yolov5/image', 10)
        self.br = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.get_logger().info("Node Initialized")
   
    def listener_callback(self, data):
        self.get_logger().info("Got Image")
        current_frame = self.br.imgmsg_to_cv2(data)
        processed_image = self.model(current_frame)
        #results = self.br.cv2_to_imgmsg(processed_image.ims[0]) # Original Img
        results = self.br.cv2_to_imgmsg(processed_image.render()[0]) # Boxed Img

        self.image_publisher.publish(results)  
       
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
   
    image_subscriber.destroy_node()
    rclpy.shutdown()
   
if __name__ == '__main__':
    main()
