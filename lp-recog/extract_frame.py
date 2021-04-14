import cv2
import os

# Function to extract frames
def FrameCapture(source, save_path):
    vidObj = cv2.VideoCapture(source)
    count = 0
    success = 1
    while True:
        success, image = vidObj.read()
        if success:
            cv2.imwrite(f"{save_path}/frame{count}.jpg", image)
        else:
            break
        count += 1
    print(count)

if __name__ == "__main__":
    source = "/content/drive/MyDrive/competitions/mosaic-r2/test_car_lp_det/P1033666.mp4"
    name = os.path.basename(source).split('.')[0]
    folder = os.path.join("/content/predictions",name)
    if os.path.isdir(folder) == False:
        os.mkdir(folder)
    save_path = os.path.join(folder,"ext_images")
    print(save_path)
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    FrameCapture(source,save_path)