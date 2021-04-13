#!/bin/sh

#cd to yolov5
cd /content/yolov5

#predict
SOURCE="/content/Diatom-Non-neuronal-Cognition/Dataset/30_bbg/imgs"
WEIGHTS="/content/drive/MyDrive/Bacillaria_Paradoxa/yolo_weights/30_bbg_images/exp2/tt_split_1000.pt"
CONF="0.4"

echo "Source of images : $SOURCE"
echo "Weights from : $WEIGHTS"
echo "Confindance : $CONF"

python3 /content/yolov5/detect.py --source $SOURCE --weights $WEIGHTS --conf $CONF --device "cpu"