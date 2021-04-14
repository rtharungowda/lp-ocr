import os

if __name__ == "__main__":
    root = "/content/predictions/"
    if os.path.isdir(root) == False:
        os.mkdir(root)
    
    #predict
    SOURCE="/content/drive/MyDrive/competitions/mosaic-r2/test_car_lp_det/"
    PROJECT="/content/predictions/"
    WEIGHTS="/content/drive/MyDrive/competitions/mosaic-r2/weights/yolov5_s/exp2/weights/best.pt"

    type_pred = "images"
    if type_pred == "images":
        NAME="images"
        if os.path.isdir(os.path.join(PROJECT,NAME)) == False:
            os.mkdir(os.path.join(PROJECT,NAME))

    os.system(f"sh predict_yolov5.sh {SOURCE} {PROJECT} {NAME} {WEIGHTS}")