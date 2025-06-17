## Yolo to Onnx
Yolo를 TensorRT에서 이용하기 위해서는 먼저 ONNX 파일로 변환해야 하는데 이때 NMS(Non-Maximum Supression) 처리가 포함된 형태로 변호나해야 한다.
```
pip install --ignore-installed ultralytics


yolo export model=yolov8n.pt format=onnx nms=True
```


## ERROR 처리

### sympy 에러 발생시
```
apt remove python3-sympy -y
```

### No module named 'onnx'
```
pip install onnx==1.9.0
```
### Downgrade the protobuf package to 3.20.x or lower.
```
pip uninstall protobuf -y
pip install protobuf==3.19.0
```classification외한 다른 모델은 모두 nms가 포함되어 있다
