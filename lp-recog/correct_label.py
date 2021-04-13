import glob

def rename(path):
    files = glob.glob(path+"/*.txt")
    # files = ['/content/drive/MyDrive/competitions/mosaic-r2/license_plate_yolov5/valid/labels/Cars0_png.rf.c125d0b60039d24a93e99a84f5e7504d.txt']
    for f in files:
        with open(f,'r') as fi:
            lines = fi.readlines()
        # if len(lines)>1:
        #     print(lines)
        text = ""
        for i in range(len(lines)):
            lines[i] = '0'+lines[i][1:]
            text+= lines[i]

        with open(f,'w') as fi:
            fi.write(text)
        # print(text)

if __name__ == "__main__":
    rename("/content/drive/MyDrive/competitions/mosaic-r2/license_plate_yolov5/valid/labels")
    rename("/content/drive/MyDrive/competitions/mosaic-r2/license_plate_yolov5/train/labels")