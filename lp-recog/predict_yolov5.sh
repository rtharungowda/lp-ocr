#!/bin/sh

#cd to yolov5
cd /content/yolov5

#predict
SOURCE=$1
PROJECT=$2
NAME=$3
WEIGHTS=$4
CONF="0.4"

echo "Source of images : $SOURCE"
echo "Weights from : $WEIGHTS"
echo "Confindance : $CONF"

python3 /content/yolov5/detect.py --source $SOURCE --weights $WEIGHTS --conf $CONF --device "cpu" --save-txt --project $PROJECT --name $NAME --exist-ok