import cv2
import PIL
import os
import time
import numpy as np
import PIL.Image as pil_image
import argparse
from test_ARCNN import denoise
from test_SRCNN import SRCNN2
from utils import mkdir
from prepare import dowmsample, jpeg, interpolation

parser = argparse.ArgumentParser()
parser.add_argument("--test-dir", type=str, default='E:\\PythonCode\\bishe\\baseline\\test_bmp_512') ####
parser.add_argument("--SR-scale", type=int, default=3)
parser.add_argument("--JPEG-factor", type=int, default=40)

args = parser.parse_args()
print(args)

LRHQ_dir = os.getcwd() + '/test_LRHQ/'
mkdir(LRHQ_dir)
LRLQ_dir = os.getcwd() + '/test_LRLQ/'
mkdir(LRLQ_dir)

test_list = os.listdir(args.test_dir)
for num, im_name in enumerate(test_list):
    start_time = time.time()
    outputs_dir = os.getcwd() + '\\output\\'
    output_dir = os.path.join(outputs_dir, im_name[:-4])
    mkdir(output_dir)
    im_file = os.path.join(args.test_dir, im_name)
    im_GroundTrue = cv2.imread(im_file, -1) # 512
    im_LRHQ = dowmsample(im_GroundTrue, args.SR_scale, LRHQ_dir, im_name.replace('.bmp','.jpg')) # 512 -> 170
    im_LRLQ = jpeg(im_LRHQ, args.JPEG_factor, LRLQ_dir, im_name) # jpeg fuction: no need to replace im_name here

    # output1
    im_LRRLQ = interpolation(im_LRLQ, args.SR_scale) # numpy
    output1_path = os.path.join(output_dir, im_name.replace('.bmp','_output1.jpg'))
    ###
    cv2.imwrite(output1_path, im_LRRLQ)

    # output2
    #### TODO:change the channel可能是这个
#####
    img_format = '.jpg' # jpg or png
    ###
    output2_name = im_name.replace('.bmp', ('_x{}_output2.jpg').format(args.SR_scale))
    output2_path = os.path.join(output_dir,output2_name)
    ###
    output2 = SRCNN2(args, output1_path)
    output2.save(output2_path)
    ### TODO:没有import Image, only SRCNN using PIL

    # output3
    OutputTemp_name = im_name.replace('.bmp','_LRHQQ.png')
    OutputTemp_path = os.path.join(output_dir, OutputTemp_name)
    im_LRLQ_file = os.path.join(LRLQ_dir, im_name.replace('.bmp','.jpg'))
    im_LRLQ2 = cv2.imread(im_LRLQ_file, -1)
    LRHQQ = denoise(args, im_LRLQ2, OutputTemp_path) # im_LRLQ: numpy
    height = np.size(LRHQQ, 0)
    width = np.size(LRHQQ, 1)
    im_LRRHQQ = cv2.resize(LRHQQ, (int(height*args.SR_scale), int(width*args.SR_scale)), interpolation=cv2.INTER_CUBIC)
    OutputTemp_name2 = im_name.replace('.bmp','_LRRHQQ.png'.format(args.JPEG_factor))
    OutputTemp_path2 = os.path.join(output_dir, OutputTemp_name2)
    cv2.imwrite(OutputTemp_path2,im_LRRHQQ)
    # TODO: STORE THE im_LRRHQQ and read, then input the new SRCNN2()
    # convert_to_PIL class
    im_LRRHQQ_PIL = pil_image.fromarray(im_LRRHQQ)
    output3 = SRCNN2(args, OutputTemp_path2)
    output3_name = im_name.replace('.bmp', ('_arcnn_{}'+'_srcnn_x{}_output3'+'.png').format(args.JPEG_factor, args.SR_scale))
    output3_path = os.path.join(output_dir, output3_name)
    #
    output3.save(output3_path)

    end_time = time.time()
    per_time = end_time-start_time
    print('image {}, time_consume={}'.format(num, per_time))


    # # data analysis
    # psnr = calc_psnr(y, preds)
    # print('PSNR: {:.2f}'.format(psnr))
    # # TODO:





