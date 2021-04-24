import cv2
import numpy as np
import sys, os
import imutils
from skimage.filters import threshold_local
from skimage import measure

def contrast(img: np.ndarray) -> np.ndarray:
    """Improves contrast to deal with blurry images

    Args:
        img (np.ndarray): Image

    Returns:
        img2 (np.ndarray): high-contrast image
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe=cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l,a,b=cv2.split(lab)  # split on 3 different channels

    l2=clahe.apply(l)  # apply CLAHE to the L-channel

    lab=cv2.merge((l2,a,b))  # merge channels
    img2=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

def resize_with_aspect(image: np.ndarray, width: int = 500) -> np.ndarray:
    """Resize the image while preserving the aspect ratio

    Args:
        image (np.ndarray): Image to be resized
        width (int, optional): Width to be resized to. Defaults to 500.

    Returns:
        np.ndarray: Resized Image
    """
    height = int(width/image.shape[1] * image.shape[0])
    return cv2.resize(image, (width, height))

def clean_plate(cvImage: np.ndarray, fixed_width: int) -> np.ndarray:
    """
    Extract Value channel from the HSV format of image and apply adaptive thresholding
    to reveal the characters on the license plate.
    """
    plate_img = cvImage.copy()
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    thresh = cv2.bitwise_not(thresh)

    """At this point, we tried applying the tried and tested method of opening
    and closing the image using erosion and dilation, which usually works well for removing noise.
    However, in this case, it was also separating characters like 'N', which a thinner connection
    between the 2 vertical lines. Thus, removing noise like this became a risk which would not pay off.
    We found segmenting the characters based on size to be a better approach than this, but this could
    be feasible if the letters and numbers were clear in all the images."""

    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    # Makes the letters slightly thicker
    kernel = np.ones((3, 3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    return thresh

def perform_segmentation(image: np.ndarray):
    """Performs Segmentation

    Args:
        image (np.ndarray): Image to be segmented

    Returns:
        list(np.ndarray): List of segmented images
    """
    # Resizes the image, keeping the width fixed at 500
    image = resize_with_aspect(image, 500)

    # Increases contrast of the image using the CLAHE technique
    image = contrast(image)

    # Removes noise from coloured images
    
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    """For PS1, we found out the skew angle by drawing a slanted rectangle as a bounding box,
    around the largest detected contour of the image.
    This approach fails for number plate detection as we are unable to draw the bounding rectangle
    around the text due to the presence of the number plate itself."""
    # angle = getSkewAngle(image)
    # new_img = rotateImage(image, -1.0 * angle)
    # new_img = resize_with_aspect(new_img, 500)


    new_img = image


    iwl_bb = clean_plate(new_img, 500)
    iwl_wb = cv2.bitwise_not(iwl_bb)
    # cv2.imshow("iwl_bb", iwl_bb)
    # cv2.imshow("iwl_wb", iwl_wb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Total area of the resized image.
    img_area = iwl_bb.shape[1] * iwl_bb.shape[0]


    """Here we had the option of using RETR_EXTERNAL (or) RETR_TREE"""
    contours,_=cv2.findContours(iwl_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rectangles=[]

    """We have only selected contours that are of a size bigger that 1/200th Area of the image,
    or are not bigger than half the image size.
    A further classification of height not less that 1/5th of the image height, and
    width not more than 1/5th of the image width was also added to remove the inherent noise
    in the given number plate."""
    for cnt in contours:
        if cv2.contourArea(cnt) > img_area / 2:
            continue
        if cv2.contourArea(cnt) > img_area / 200:
            x,y,w,h=cv2.boundingRect(cnt)
            if(h < (iwl_bb.shape[0] / 5) or w > (iwl_bb.shape[1] / 5)):
                continue
            rectangles.append([x,y,w,h])

    "Removing cases where a contour is inside another contour like in 'O' and 'D'"
    final_rect = []
    for (x, y, w, h) in rectangles:
        flag = True
        for(x2, y2, w2, h2) in rectangles:
            if x > x2 and y > y2 and x + w < x2 + w2 and y + h < y2 + h2:
                flag = False
                break
        if flag:
            final_rect.append([x,y,w,h])

    rectangles = final_rect


    rectangles.sort()
    print("Number of characters recognized:", len(rectangles))
    

    images = []

    """This part segments each image from the number plate based on the rectangle 
    coordinates received previously."""
    for i in range(len(rectangles)):
        cv2.rectangle(iwl_wb,
            (rectangles[i][0],rectangles[i][1]),
            (rectangles[i][0]+rectangles[i][2],
            rectangles[i][1]+rectangles[i][3]),
            (0,255,0),
            3
        )
        image=iwl_bb[
            rectangles[i][1] : rectangles[i][1] + rectangles[i][3],
            rectangles[i][0] : rectangles[i][0] + rectangles[i][2]
        ]
        larger = max(image.shape[0], image.shape[1])
        smaller = min(image.shape[0], image.shape[1])
        border=int(0.2 * larger)
        # Adds a border around the cropped image.
        image = cv2.copyMakeBorder(
            image,
            top = (larger - smaller) // 2 + border,
            bottom = (larger - smaller) // 2 + int( 1 * border),
            left = border,
            right = int(1 * border),
            borderType = cv2.BORDER_CONSTANT,
            value = [0, 0, 0])
        
        # cv2.namedWindow("img_{}".format(i), cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("img_{}".format(i), image)
        images.append(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return images


if __name__ == "__main__":
    img_path = "/home/tharun/Downloads/lp-ocr/submission/predictions/video/segmented/video_3_0.jpg"
    image = cv2.imread(img_path)
    perform_segmentation(image)
