### Yolov8 (1,84,8400)

의미는 (batch, num_classes + 4(xywh), num_anchors)

### Yolov8 (1,300,6)

의미는 (batch, num_detections, 6)

#### 1 (Batch Size):

* 첫 번째 차원은 배치(Batch) 크기를 나타냅니다. 즉, 한 번에 하나의 이미지를 처리하고 있다는 의미입니다.

#### 300 (Number of Detections):

* 두 번째 차원은 NMS를 거쳐 최종적으로 남아있는 탐지된 객체의 최대 개수를 나타냅니다. YOLOv8 모델은 일반적으로 NMS를 적용하기 전에 훨씬 많은 수의 예측(예: 8400개)을 생성하지만, NMS는 중복되거나 신뢰도가 낮은 박스를 제거하여 최종적으로 가장 좋은 300개의 탐지 결과만 남기는 역할을 합니다. 이 값은 모델을 ONNX로 내보낼 때 설정된 topk (최대 탐지 개수) 값에 따라 달라질 수 있습니다.

#### 6 (Detection Attributes):

* 세 번째 차원은 각 탐지된 객체에 대한 6가지 정보를 나타냅니다. 일반적인 YOLOv8 ONNX 출력 형식에서 이 6가지 정보는 다음과 같습니다:

    1. x1: 바운딩 박스의 좌상단 x 좌표

    2. y1: 바운딩 박스의 좌상단 y 좌표

    3. x2: 바운딩 박스의 우하단 x 좌표

    4. y2: 바운딩 박스의 우하단 y 좌표

    5. confidence (또는 score): 해당 바운딩 박스가 객체를 포함할 확률 및 해당 객체의 클래스 예측 정확도 (객체 존재 신뢰도와 클래스  신뢰도가 결합된 값)

    6. class_id (또는 label): 탐지된 객체의 클래스 ID (예: 0 for person, 1 for car 등)

### Check Memory Leak

```
sudo apt-get install valgrind
gcc -g -o main.out main.c
valgrind --leak-check=yes ./main.out
'''
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
