#!/bin/sh

#cd to yolov5
cd /content/yolov5

#predict
SOURCE="/content/drive/MyDrive/competitions/mosaic-r2/test_car_lp_det"
WEIGHTS="/content/drive/MyDrive/competitions/mosaic-r2/weights/yolov5_s/exp2/weights/best.pt"
CONF="0.4"

echo "Source of images : $SOURCE"
echo "Weights from : $WEIGHTS"
echo "Confindance : $CONF"

python3 /content/yolov5/detect.py --source $SOURCE --weights $WEIGHTS --conf $CONF --device "cpu"