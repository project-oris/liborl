# model description

### 1. yolo11n_nms.onnx 

Yolo11n Detection Model

### 2. yolo11n-cls.onnx  

Yolo11n Classification Model

### 3. yolo11n-obb.onnx

Yolo11n Oriented Bounding Box Detection Model

### 4. yolo11n-pose.onnx

Yolo11n Pose Estimation Model

### 5. yolo11n-seg.onnx

Yolo11n Segementation Model

* input : 

    - "image" : (1,3,640,640)

* output: 

    - "output0" : (1,300,38)