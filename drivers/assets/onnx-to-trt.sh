if [ $# -lt 1 ]; then
	echo "Usage: $0 onnx_file" 
        exit -1
else
FILE_NAME=$(basename $1)
FILE_DIR=$(dirname $1)
FILE_NAME_ONLY=$(echo $FILE_NAME | cut -d. -f1)
NEW_FILE_PATH="$FILE_DIR/$FILE_NAME_ONLY.engine"
    trtexec --onnx=$1 --saveEngine=$NEW_FILE_PATH
fi
