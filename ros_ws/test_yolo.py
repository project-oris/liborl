from typing import List, Tuple

import cv2
import time
import numpy as np
import sys
from yolo_detect.yolo_detect_module import (
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

class YOLO:
    def __init__(self, config_path: str):
        self.config_path = config_path        
        self.classes = class_names
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.driver = load_driver(config_path)

        input_shape = get_string_option(self.driver, "tensor/input_shape").strip('()').split(',')
        shape_list = list(map(int, input_shape))        

        self.input_height = shape_list[0]
        self.input_width = shape_list[1]
        self.normalization = get_bool_option(self.driver, "tensor/normalization")
        self.normalization_base = get_float_option(self.driver, "tensor/normalization_base")
        self.tochw = get_bool_option(self.driver, "yolo/tochw")
        
        print("tochw ", self.tochw)
        print("normalization ", self.normalization)
        print("normalization_base ", self.normalization_base)       
        

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

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

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # Read the input image using OpenCV
        self.img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        print("image org [",self.img_width, ",", self.img_height,"]")

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
        for r in results:
            class_id = int(r[[5]])
            x1 = min(int(r[0] * ratio), input_image.shape[1])
            y1 = min(int(r[1] * ratio), input_image.shape[0])
            x2 = min(int(r[2] * ratio), input_image.shape[1])
            y2 = min(int(r[3] * ratio), input_image.shape[0])   
            c_score = r[4]

            if c_score >0:            
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

        return input_image

    def infer(self, img) -> np.ndarray:         

        img_data, snap_data, ratio = self.preprocess(img)

        results = run_infer(self.driver, [img_data])        

        for r in results:
            print("result: ", r.shape)            
            for rr in r:
                if rr[4] > 0.5:
                    print("value:", rr[:4] * ratio, "..", rr[4], "..", int(rr[5]))

        #out_img = self.postprocess(self.img, results[0], ratio)
        out_img = self.postprocess(snap_data, results[0], 1.0)

        cv2.imshow("Output", out_img)
        cv2.waitKey(10)
        


        # for i, r in enumerate(output):
        #     print(f"Result {i}: shape = {r.shape}, dtype = {r.dtype}")
        #     print(r)  # 혹은 일부 출력: print(r[0, :5])        



video_path = "./assets/input1.mp4"
cap = cv2.VideoCapture(video_path)
detection = YOLO("./config.json")

while True:
    retval, frame = cap.read()
    if not(retval):
        break
    start = time.time()
    detection.infer(frame)
    end = time.time()
    print("ELAPSED : ", end-start)





#t = makeTensor(DataType.FLOAT32, [1, 3, 224, 224])

#print("Shape:", t.shape)
#print("DataType:", t.data_type)

# m = Driver()
#m = load_driver("./config.json")

#print(get_input_shape(m))
#print(get_output_shape(m))

