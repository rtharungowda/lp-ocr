import os

WEIGHTS="/content/drive/MyDrive/competitions/mosaic-r2/weights/yolov5_s/exp2/weights/best.pt"
CONF="0.4"

def img(source):
    root = "/content/predictions/"
    if os.path.isdir(root) == False:
        os.mkdir(root)

    file_name = os.path.basename(source).split('.')[0]
    PROJECT = os.path.join("/content/predictions",file_name)

    if os.path.isdir(project) == False:
        os.mkdir(PROJECT)

    NAME = "yolo_detection"

    print(PROJECT,NAME)
    
    os.system(f"python3 detect.py --source {source} --weights {WEIGHTS} --conf {CONF} --device 'cpu' --save-txt --project {PROJECT} --name {NAME} --exist-ok")

def crop(source):
    img_name = os.path.basename(source).split('.')[0]
    img = cv2.imread(source)
    height = img.shape[0]
    width = img.shape[1]
    print(height,width)
    label_path = os.path.join(PROJECT,NAME,"labels")
    label = os.path.join(label_path,img_name+'.txt')

    seg_path = os.path.join(PROJECT,"segmented")
    

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
    source="/content/drive/MyDrive/competitions/mosaic-r2/test_car_lp_det/several2.jpeg"
    if source[-3:]=="mp4":
        pass
    else:
        img(source)
