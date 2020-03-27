import  numpy as np
import  cv2
import os
from utils import mkdir



def dowmsample(im, down_factor, LRHQ_dir, im_name): # image should be in numpy, which means using cv2 to read image
    LRHQ_path = os.path.join(LRHQ_dir, im_name)
    height = np.size(im,0)
    width = np.size(im,1)
    img_blur = cv2.GaussianBlur(im, (5, 5), 0)
    im_downsample = cv2.resize(img_blur, (height//down_factor, width//down_factor))
    cv2.imwrite(LRHQ_path, im_downsample)
    return im_downsample

def jpeg(im_downsample, JPEG_factor, LRLQ_dir, im_name):
    LRLQ_path = os.path.join(LRLQ_dir, im_name.replace('.bmp','.jpg'))
    cv2.imwrite(LRLQ_path, im_downsample, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_factor])
    im_jpeg = cv2.imread(LRLQ_path)
    return im_jpeg

def interpolation(im_jpeg, down_factor):
    height = np.size(im_jpeg,0)
    width = np.size(im_jpeg,1)
    im_interpolation = cv2.resize(im_jpeg, (int(height*down_factor), int(width*down_factor)), interpolation=cv2.INTER_CUBIC)
    return im_interpolation