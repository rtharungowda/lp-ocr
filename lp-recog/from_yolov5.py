import glob 
import os
import cv2 

LABEL_PATH = "/content/yolov5/runs/detect/exp2/labels"

def crop(path):
    img_name = os.path.basename(path).split('.')[0]
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    print(height,width)
    label = os.path.join(LABEL_PATH,img_name+'.txt')
    with open(label,'r') as fi:
        lines = fi.readlines()
    for line in lines:
        param = line.split(' ')
        x = float(param[1])*width
        y = float(param[2])*height
        w = float(param[3])*width
        h = float(param[4])*height
        print(x,y,w,h)
        start_point = (int(x-w/2),int(y-h/2))
        end_point = (int(x+w/2),int(y+h/2))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)
        cv2.imwrite(f"/content/{img_name}.jpg",img)

if __name__ == "__main__":
    path = "/content/drive/MyDrive/competitions/mosaic-r2/test_car_lp_det/OIP(31).jpeg"
    crop(path)