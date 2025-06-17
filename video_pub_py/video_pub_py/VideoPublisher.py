#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge  
import cv2

class VideoPublisher(Node):

    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter("video_path", "./assets/input1.mp4")
        self.declare_parameter("topic_name", "camera/image_raw")

        video_file_path = self.get_parameter('video_path').get_parameter_value().string_value
        if video_file_path=="":
            self.get_logger().error(f"No video file path provided. Please set the 'video_path' parameter.")
            self.destroy_node()
            return
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge_ = CvBridge()
        self.video_path = video_file_path               
        try:
            self.cap = cv2.VideoCapture(self.video_path)            
            self.frame_rate = 30 #self.meta.get('fps',30)
        except Exception as e:
            self.get_logger().error(f"Could not open video file : {self.video_path} - {e}")
            self.destroy_node()
            return

        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)

    def publish_frame(self):
        try:
            retval, frame = self.cap.read()
            if not(retval):
                return
            try:
                img_msg = self.bridge_.cv2_to_imgmsg(np.ascontiguousarray(frame), encoding="bgr8") 
                self.publisher_.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Error converting or publishing image: {e}")
        except StopIteration:
            self.get_logger().info("End of video file reached.")
            self.destroy_timer(self.timer)
            self.destroy_node()
        except Exception as e:
            self.get_logger().error(f"Error reading video frame: {e}")

def main(args=None):

    rclpy.init(args=args)    
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
