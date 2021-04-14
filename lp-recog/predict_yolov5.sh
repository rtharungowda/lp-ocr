#!/bin/sh

#cd to yolov5
cd /content/yolov5

#predict
SOURCE="/content/predictions/P1033666/ext_images"
PROJECT="/content/predictions/P1033666/yolov5"
NAME="yolov5"
WEIGHTS="/content/drive/MyDrive/competitions/mosaic-r2/weights/yolov5_s/exp2/weights/best.pt"
CONF="0.4"

echo "Source of images : $SOURCE"
echo "Weights from : $WEIGHTS"
echo "Confindance : $CONF"

python3 /content/yolov5/detect.py --source $SOURCE --weights $WEIGHTS --conf $CONF --device "cpu" --save-txt --project $PROJECT --name $NAME --exist-ok