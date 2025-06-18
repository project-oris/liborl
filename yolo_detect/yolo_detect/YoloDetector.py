from typing import List, Tuple
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge  
import cv2
import json
from oris_orl_py import (
    makeTensor, 
    DataType, 
    Driver, 
    load_driver, 
    get_output_shape, 
    get_input_shape, 
    run_infer,
    get_int_option,
    get_float_option,
    get_string_option,
    get_bool_option
)
from std_msgs.msg import String


# class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class_names = ['person', 'car', 'ev charging station', 'cctv', 'gas station', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']               

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detect')
        self.declare_parameter("config_path", "./config.json")
        self.declare_parameter("input_topic", "camera/image_raw")

        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        if config_path=="":
            self.get_logger().error(f"No config file path provided. Please set the 'config_path' parameter.")
            self.destroy_node()
            return
        input_topic_name = self.get_parameter('input_topic').get_parameter_value().string_value

        self.config_path = config_path        
        self.classes = class_names
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.driver =load_driver(config_path)
        input_shape = get_string_option(self.driver, "tensor/input_shape").strip('()').split(',')
        shape_list = list(map(int, input_shape))        
        self.input_height = shape_list[0]
        self.input_width = shape_list[1]
        self.normalization = get_bool_option(self.driver, "tensor/normalization")
        self.normalization_base = get_float_option(self.driver, "tensor/normalization_base")
        self.tochw = get_bool_option(self.driver, "yolo/tochw")
        self.bridge_ = CvBridge()

        qos = QoSProfile(depth=10)

        self.publisher_ = self.create_publisher(String, "detect_results/data", qos) # json dump string
        self.img_publisher_ = self.create_publisher(Image, "detect_results/image", qos) # json dump string
        self.img_sub = self.create_subscription(Image, input_topic_name, self.image_callback, qos)
        
    def image_callback(self, data):        
        current_frame = self.bridge_.imgmsg_to_cv2(data, "bgr8")
        self.infer(current_frame)        

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  

    def preprocess(self, img) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.
        Returns:
            (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            (np.ndarray): Image resized to display inference results.
            (float): resize ratio
        """
        # Read the input image using OpenCV
        self.img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]       

        # Convert the image color space from BGR to RGB
        c_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # c_img =  img        

        width = self.input_width 
        height = self.input_height 

        r = min(width / (c_img.shape[1] * 1.0), height / (c_img.shape[0] * 1.0))

        unpad_w = int(r * c_img.shape[1])
        unpad_h = int(r * c_img.shape[0])
        bgcolor = (0, 0, 0)

        #t_img = cv2.resize(c_img, (unpad_w, unpad_h))
        re = cv2.resize(c_img, (unpad_w, unpad_h))
        t_img = re

        #re = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)

        converted = np.full((height, width, 3), bgcolor, dtype=np.uint8)
        converted[0:re.shape[0], 0:re.shape[1]] = re

        ratio = 1.0 / min(width / float(c_img.shape[1]), height / float(c_img.shape[0]))            

        return converted, t_img, ratio

    def postprocess(self, input_image: np.ndarray, results: List[np.ndarray], ratio: float) -> np.ndarray:
        # Get the number of rows in the outputs array  
        results_data = []
        
        for r in results:
            c_score = r[4]
            class_id = int(r[[5]])

            if c_score >0 and class_id >=0 and class_id < len(self.classes):
                fx1 = float(min(abs(int(r[0] * ratio)), self.img_width)) / float(self.img_width) 
                fy1 = float(min(abs(int(r[1] * ratio)), self.img_height)) / float(self.img_height)
                fx2 = float(min(abs(int(r[2] * ratio)), self.img_width)) / float(self.img_width) 
                fy2 = float(min(abs(int(r[3] * ratio)), self.img_height)) / float(self.img_height)

                x1 = int(fx1 * input_image.shape[1])
                y1 = int(fy1 * input_image.shape[0])
                x2 = int(fx2 * input_image.shape[1])
                y2 = int(fy2 * input_image.shape[0])

                result_item = { "class_id" : self.classes[class_id], 
                    "x1" :  round(fx1, 7),  
                    "y1" :  round(fy1, 7), 
                    "x2" :  round(fx2, 7),
                    "y2" :  round(fy2, 7), 
                    "confidence" : round(float(c_score), 7)}

                results_data.append(result_item)

                color = self.color_palette[class_id]
                cv2.rectangle(input_image, (x1,y1), (x2,y2), color, 2)
                label = f"{self.classes[class_id]}: {c_score:.2f}"

                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_x = x1
                label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
                cv2.rectangle(
                input_image, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
                )

                cv2.putText(input_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)            

        return results_data, input_image

    def infer(self, img):
        img_data, snap_data, ratio = self.preprocess(img)
        results = run_infer(self.driver, [img_data])                
        if len(results) ==0:
            return
        resultData, out_img = self.postprocess(snap_data, results[0], ratio)

        data_msg = String()
        data_msg.data = json.dumps(resultData)
        print(data_msg.data)
        self.publisher_.publish(data_msg)
        img_msg = self.bridge_.cv2_to_imgmsg(np.ascontiguousarray(out_img), encoding="rgb8") 
        self.img_publisher_.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)    
    yolo_detector = YoloDetector()
    rclpy.spin(yolo_detector)
    yolo_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()        