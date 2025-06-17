from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Process pt file.')
parser.add_argument('--pt_path', help='path to pt file', required=True)
args = parser.parse_args()
# TODO: Specify which model you want to convert
# Model can be downloaded from https://github.com/ultralytics/ultralytics
model = YOLO(args.pt_path)
model.fuse()
model.info(verbose=False)  # Print model information

model.export(format="onnx", opset=17, nms=True, simplify=True)  ## If use TensorRT 8.6.1 , opset has to be 17
#model.export(format="onnx", opset=17, simplify=True)  ## If use TensorRT 8.6.1 , opset has to be 17
