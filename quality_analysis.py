from psnr import  PSNR
import argparse
import os
from utils import  mkdir
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--test-dir", type=str, default='E:\\PythonCode\\bishe\\baseline\\test_bmp_512') ####
parser.add_argument("--SR-scale", type=int, default=2)
parser.add_argument("--JPEG-factor", type=int, default=20)

args = parser.parse_args()
print('args: ', args)

test_list = os.listdir(args.test_dir)
lists = list()
for num, im_name in enumerate(test_list):
    print('calculating the psnrs of image: ', im_name)
    outputs_dir = os.getcwd() + '\\output\\'
    GroundTrue_path = os.path.join(os.getcwd() + '\\test_bmp_512\\', im_name)
    im_name = im_name[:-4] # the im_name has no format
    output_dir = os.path.join(outputs_dir, im_name)
    mkdir(output_dir)
    output1_path = os.path.join(output_dir, im_name+'_output1.jpg')
    output2_path = os.path.join(output_dir, im_name+'_x2_output2.jpg')
    output3_path = os.path.join(output_dir, im_name+'_arcnn_20_srcnn_x2_output3.png') # TODO: change the scales
    required_width = 512; required_height = 512
    psnr1 = PSNR(GroundTrue_path, output1_path, required_width, required_height)
    psnr2 = PSNR(GroundTrue_path, output2_path, required_width, required_height)
    psnr3 = PSNR(GroundTrue_path, output3_path, required_width, required_height)
    print('psnr1: ', psnr1, '|psnr2: ', psnr2, '|psnr3: ', psnr3)
    psnr_list = [psnr1, psnr2, psnr3]
    lists.append(psnr_list)
PSNR_Array = np.array(lists)
print(PSNR_Array)
PSNR_CSV = os.path.join(outputs_dir,'PSNR_arcnn_20_srcnn_x2.csv')# TODO:
np.savetxt(PSNR_CSV,PSNR_Array,fmt='%f', delimiter = ',')

