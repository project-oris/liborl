if [ $# -lt 1 ]; then
	echo "Usage: $0 torch_file_name" 
        exit -1
else
  #yolo export model=$1 format=onnx nms=True
  python3 pytorch2onnx.py --pt_path $1
fi
