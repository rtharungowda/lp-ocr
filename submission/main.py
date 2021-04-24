import os
import cv2
import shutil
import glob
import numpy as np

from segmentation import perform_segmentation
from predict import predict_charac

REV_MAPPING = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
    16:'G',
    17:'H',
    18:'I',
    19:'J',
    20:'K',
    21:'L',
    22:'M',
    23:'N',
    24:'P',
    25:'Q',
    26:'R',
    27:'S',
    28:'T',
    29:'U',
    30:'V',
    31:'W',
    32:'X',
    33:'Y',
    34:'Z',
}

CONF="0.4"
PROJECT=""
NAME=""

def detect_lp(source):
    """detect license plate using yolo
    
    Args:
        source (str) : path to source image or video
    
    Returns:
        None
    """
    global PROJECT, NAME
    #create predictions folder
    root = os.path.join(os.path.dirname(__file__),'predictions')
    if os.path.isdir(root)==False:
        os.mkdir(root)

    #get file name for creating folder
    file_name = os.path.basename(source).split('.')[0]
    PROJECT = os.path.join(root,file_name)

    if os.path.isdir(PROJECT) == True:
        shutil.rmtree(PROJECT, ignore_errors=True)  
    os.mkdir(PROJECT)

    #yolov5 parameters
    NAME = "yolo_detection"
    print(__file__)
    pth = os.path.abspath("detect.py")
    WEIGHTS = os.path.abspath("lp_detect.pt")
    print(WEIGHTS)
    
    #execute yolov5 
    os.system(f"python3 {pth} --source {source} --weights {WEIGHTS} --conf {CONF} --device 'cpu' --save-txt --project {PROJECT} --name {NAME} --exist-ok")

def end_to_end(source):
    """detect license plate and recongnise using yolo (end to end model, not preferred)
    
    Args:
        source (str) : path to source image or video
    
    Returns:
        None
    """
    #get file names and locations
    global PROJECT, NAME
    root = os.path.join(os.path.dirname(__file__),'predictions')
    if os.path.isdir(root)==False:
        os.mkdir(root)

    file_name = os.path.basename(source).split('.')[0]
    PROJECT = os.path.join(root,file_name)

    if os.path.isdir(PROJECT) == True:
        shutil.rmtree(PROJECT, ignore_errors=True)  
    os.mkdir(PROJECT)

    #execute yolo
    NAME = "yolo_end_to_end"
    print(__file__)
    pth = os.path.abspath("detect.py")
    WEIGHTS = os.path.abspath("end_to_end.pt")
    print(WEIGHTS)
    os.system(f"python3 {pth} --source {source} --weights {WEIGHTS} --conf {CONF} --device 'cpu' --save-txt --project {PROJECT} --name {NAME} --exist-ok")

    #get label name
    img_name = os.path.basename(source).split('.')[0]
    label_path = os.path.join(PROJECT,NAME,"labels")
    label = os.path.join(label_path,img_name+'.txt')

    #load image
    image = cv2.imread(source)
    height = image.shape[0]
    width = image.shape[1]
    
    #check if any lp found
    if os.path.isfile(label) == False:
        print(label)
        print('*'*10)
        print("no license plate found")
        return 
    
    #read predictions
    with open(label,'r') as fi:
        lines = fi.readlines()
    
    #crop characters and save predictions
    for i,line in enumerate(lines):
        param = line.split(' ')
        x = float(param[1])*width
        y = float(param[2])*height
        w = float(param[3])*width
        h = float(param[4])*height
        start_point = (int(x-w/2),int(y-h/2))
        end_point = (int(x+w/2),int(y+h/2))
        img = image[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        print(param[0])
        cv2.imwrite(f"{label_path}/{i}_{REV_MAPPING[int(param[0])]}.jpeg",img)

def from_video(source):
    """get license plate bounding box from yolo text output on video
    
    Args:
        source (str) : path to video

    Returns:
        None
    """
    label_path = os.path.join(PROJECT,NAME,"labels")
    label_txts = glob.glob(label_path+"/*.txt")

    #frames in which lp was detected
    frames = []
    for txt in label_txts:
        frames.append(int(os.path.basename(txt).split('_')[-1].split('.')[0]))
    frames.sort()
    print(f"license plate detected in\n",frames)

    #read video
    vidObj = cv2.VideoCapture(source)
    video_name = os.path.basename(source).split('.')[0]

    #get predictions save path
    seg_path = os.path.join(PROJECT,"segmented")
    if os.path.isdir(seg_path)==True:
        shutil.rmtree(seg_path, ignore_errors=True)
    os.mkdir(seg_path)

    count = 1
    success = 1
    while True:
        success, image = vidObj.read()
        if success == 0:
            break
        #if frame has license plate then 
        elif count in frames:
            #get label path
            label_path = os.path.join(PROJECT,NAME,"labels")
            img_name = video_name+f'_{str(count)}'
            label_name = img_name+'.txt'
            label = os.path.join(label_path,label_name)
            crop(image, label, seg_path, img_name)
        count+=1

def from_image(source):
    """get license plate bouding box from yolo text output on image
    
    Args:
        source (str) : path to image
    
    Returns:
        None
    """
    #get label path 
    img_name = os.path.basename(source).split('.')[0]
    image = cv2.imread(source)
    label_path = os.path.join(PROJECT,NAME,"labels")
    label = os.path.join(label_path,img_name+'.txt')
    #save predictions path
    seg_path = os.path.join(PROJECT,"segmented")
    if os.path.isdir(seg_path)==True:
        shutil.rmtree(seg_path, ignore_errors=True)
    os.mkdir(seg_path)

    crop(image, label, seg_path, img_name)

def crop(image, label, seg_path, img_name):
    """crops, segments and predicts license plate characters
    
    Args:
        image (np.array) : cv2 image array
        label (str) : path to text file
        seg_path (str) : path to save segmented image
        img_name (str) : source image/video frame name
    
    Returns:
        None
    """
    #check if label is present
    if os.path.isfile(label) == False:
        print(label)
        print('*'*10)
        print("no license plate found")
        return 
    
    with open(label,'r') as fi:
        lines = fi.readlines()

    
    height = image.shape[0]
    width = image.shape[1]
    
    for i,line in enumerate(lines):
        param = line.split(' ')
        #get coordinates from prediction
        x = float(param[1])*width
        y = float(param[2])*height
        w = float(param[3])*width
        h = float(param[4])*height
        
        #find start and end points of bounding box
        start_point = (int(x-w/2),int(y-h/2))
        end_point = (int(x+w/2),int(y+h/2))
        
        #crop image
        img = image[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        save_path = os.path.join(seg_path,f"{img_name}_{i}.jpg")
        cv2.imwrite(save_path,img)
        cv2.imshow("cropped",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #-----------
        #get segmented images
        images = perform_segmentation(img)
        for i,img_ in enumerate(images):
            cv2.imshow("character",img_)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #make predictions
            label, mapping = predict_charac(img_)
            print(mapping)
        #-----------

def PlateRecognition(source, detect_lp_also=False):
    """plate recongition

    Args:
        source (str): path to image file or video
        detect_lp_also (bool, optional): detect license plate also from the image or video, then perform ocr. Defaults to False.
    """
    #perform segementation and classification only
    #given image containes only license plate
    if detect_lp_also == False:
        image = cv2.imread(source)
        cv2.imshow("image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        images = perform_segmentation(image)
        for img in images:
            label, mapping = predict_charac(img)
            print(mapping)
    else:
        detect_lp(source)#detect license plate in image/video
        if source[-3:]=="mp4" or source[-3:]=="MOV":#if video 
            from_video(source)
        else:
            from_image(source)
        print()
        print("*"*10)
        print("-"*10)
        print("results saved at")
        print(f"yolov5 predictions saved at {PROJECT}/{NAME} and labels at {PROJECT}/{NAME}/labels")
        print(f"segmented and recognised license plate ocr saved at {PROJECT}/segmented")

if __name__ == "__main__":

    source = glob.glob("/home/tharun/Downloads/PS2-20210418T160413Z-001/PS2/test_pics"+'/*.png')
    for f in source:
        print("now predicting ",f)
        detect_lp_also = False #change to true if u want to detect the license plate also
        
        #segmentatino and then classification
        PlateRecognition(f, detect_lp_also)
        
        #end to end prediction
        # end_to_end(f)
    
    
