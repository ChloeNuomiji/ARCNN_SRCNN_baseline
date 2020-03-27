import torch
import numpy as np
import PIL.Image as pil_image
import cv2
import os
from utils import AverageMeter, calc_psnr, get_torch_y

def PSNR(img1_path, img2_path, required_width, required_height):
    img1_y = get_torch_y(img1_path, required_width, required_height)
    img2_y = get_torch_y(img2_path, required_width, required_height)
    psnr = calc_psnr(img1_y, img2_y)
    return psnr



