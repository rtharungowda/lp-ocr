
# License plate detection, segmentation and recongnition

## File description

+ `detect.py` - performs license plate detection in given image
+ `predict.py` - perfroms character recognition on segmented images
+ `segmentation.py` - performs character segmentation on license plates
+ `main.py` - the file which the takes in the source path and performs all operations by calling other 

## Predictions:

+ The predictions(from yolov5)are stored in a folder called predicitons.
+ The file directory is as follows:
    + ├── submission </br>
        + ├── predictions </br>
            + ├── image/video_name </br>
                + ├── segmented </br>
                    + ... -> images cropped contains only lp </br>
                + ├── yolo_detection </br>
                    + ... -> images with bounding box around lp </br>
                    + ├── labels </br>
                        + .... -> labels in .txt files </br>
        + ├──utils </br>
        + ├──models </br>
.......
