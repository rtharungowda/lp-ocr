import os
import cv2
import shutil

WEIGHTS="/content/drive/MyDrive/competitions/mosaic-r2/weights/yolov5_s/exp2/weights/best.pt"
CONF="0.4"
PROJECT=""
NAME=""

def img(source):
    global PROJECT, NAME
    root = "/content/predictions/"
    if os.path.isdir(root) == False:
        os.mkdir(root)

    file_name = os.path.basename(source).split('.')[0]
    PROJECT = os.path.join("/content/predictions",file_name)

    if os.path.isdir(PROJECT) == True:
        shutil.rmtree(PROJECT, ignore_errors=True)  
    os.mkdir(PROJECT)

    NAME = "yolo_detection"

    # print(PROJECT,NAME)
    
    os.system(f"python3 detect.py --source {source} --weights {WEIGHTS} --conf {CONF} --device 'cpu' --save-txt --project {PROJECT} --name {NAME} --exist-ok")

    crop(source)

def crop(source):
    img_name = os.path.basename(source).split('.')[0]
    image = cv2.imread(source)
    height = image.shape[0]
    width = image.shape[1]
    # print(height,width)
    label_path = os.path.join(PROJECT,NAME,"labels")
    label = os.path.join(label_path,img_name+'.txt')

    seg_path = os.path.join(PROJECT,"segmented")
    if os.path.isdir(seg_path)==True:
        shutil.rmtree(seg_path, ignore_errors=True)
    os.mkdir(seg_path)

    # print(label)
    with open(label,'r') as fi:
        lines = fi.readlines()

    # print(lines)
    
    for i,line in enumerate(lines):
        param = line.split(' ')
        x = float(param[1])*width
        y = float(param[2])*height
        w = float(param[3])*width
        h = float(param[4])*height
        # print(x,y,w,h)
        start_point = (int(x-w/2),int(y-h/2))
        end_point = (int(x+w/2),int(y+h/2))
        # img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
        img = image[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        # print(img.shape)

        #-----segementation----
        # img = segmentation(img)
        #------
        save_path = os.path.join(seg_path,f"lp_{i}.jpg")
        cv2.imwrite(save_path,img)

if __name__ == "__main__":
    source="/content/drive/MyDrive/competitions/mosaic-r2/test_car_lp_det/several2.jpeg"
    if source[-3:]=="mp4":
        pass
    else:
        img(source)
