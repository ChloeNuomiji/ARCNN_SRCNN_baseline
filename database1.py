import cv2
import PIL
import os
import time
import numpy as np
import PIL.Image as pil_image
import argparse
from test_ARCNN import denoise
from test_SRCNN import SRCNN2
from utils import mkdir, PSNR, AverageMeter
from prepare import dowmsample, jpeg, interpolation

# input the test image with scales
# output three output image


def ARCNN_SRCNN(args, required_width, required_height, output1_MeanPSNR,
                        output2_MeanPSNR, output3_MeanPSNR, MeanTime,
                        PatchName_List, PSNR1_List, PSNR2_List, PSNR3_List,
                        TimeConsume_List):
    test_list = os.listdir(args.TestPatch_dir)
    for num, im_name in enumerate(test_list[:2]): ###这是输入im
        start_time = time.time()
        outputs_dir = os.getcwd() + '\\output\\JPEG_{} SR_X{}\\'.format(args.JPEG_factor, args.SR_scale)
        output_dir = os.path.join(outputs_dir, im_name[:-4])
        mkdir(output_dir)
        im_LRHQ_name = im_name.replace('.bmp', '_LRHQ.bmp')
        im_LRLQ_name = im_name.replace('.bmp', '_LRLQ.jpg')
        im_path = os.path.join(args.TestPatch_dir, im_name)
        im_GroundTrue = cv2.imread(im_path, -1) # 512
        im_LRHQ = dowmsample(im_GroundTrue, args.SR_scale, output_dir, im_LRHQ_name) # 512 -> 170
        im_LRLQ = jpeg(im_LRHQ, args.JPEG_factor, output_dir, im_LRLQ_name) # jpeg fuction: no need to replace im_name here

        # output1
        im_LRRLQ = interpolation(im_LRLQ, args.SR_scale) # numpy
        output1_path = os.path.join(output_dir, im_name.replace('.bmp','_output1.jpg'))
        cv2.imwrite(output1_path, im_LRRLQ)
        psnr1 = PSNR(im_path, output1_path, required_width, required_height)
        output1_MeanPSNR.update(val=psnr1, n=1)

        # output2
        #### TODO:change the channel可能是这个
    #####
        img_format = '.jpg' # jpg or png
        ###
        output2_name = im_name.replace('.bmp', ('_output2.jpg').format(args.SR_scale))
        output2_path = os.path.join(output_dir,output2_name)
        ###
        output2 = SRCNN2(args, output1_path)
        output2.save(output2_path)
        psnr2 = PSNR(im_path, output2_path, required_width, required_height)
        output2_MeanPSNR.update(val=psnr2, n=1)
        ### TODO:没有import Image, only SRCNN using PIL

        # output3
        OutputTemp_name = im_name.replace('.bmp','_LRHQQ.png')
        OutputTemp_path = os.path.join(output_dir, OutputTemp_name)
        im_LRLQ_file = os.path.join(output_dir, im_LRLQ_name)
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
        output3_name = im_name.replace('.bmp', ('_output3'+'.png').format(args.JPEG_factor, args.SR_scale))
        output3_path = os.path.join(output_dir, output3_name)
        #
        output3.save(output3_path)
        psnr3 = PSNR(im_path, output3_path, required_width, required_height)
        output3_MeanPSNR.update(val=psnr3, n=1)

        end_time = time.time()
        MeanTime.update(val=end_time-start_time, n=1)
        PatchName_List.append(im_name[:-4])
        PSNR1_List.append(float(psnr1))
        PSNR2_List.append(float(psnr2))
        PSNR3_List.append(float(psnr3))
        TimeConsume_List.append(end_time-start_time)

        # print('image {}, time_consume={}'.format(num, per_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--TestPatch-dir", type=str, default=r'E:\database\test')  ####
    parser.add_argument("--SR-scale", type=int, default=2)
    parser.add_argument("--JPEG-factor", type=int, default=20)

    args = parser.parse_args()
    print(args)
    required_width=255
    required_height=255
    output1_MeanPSNR = AverageMeter()
    output2_MeanPSNR = AverageMeter()
    output3_MeanPSNR = AverageMeter()
    MeanTime = AverageMeter()
    TestFullImage_list = sorted(os.listdir(args.TestPatch_dir))
    TestFullbmpImage_list = [file for file in TestFullImage_list if file.endswith('.bmp')]
    print(TestFullbmpImage_list)
    TestPatch_dir = args.TestPatch_dir
    PatchName_List = list()
    PSNR1_List = list()
    PSNR2_List = list()
    PSNR3_List = list()
    TimeConsume_List = list()
    for num, TestFullImage in enumerate(TestFullbmpImage_list):
            args.TestPatch_dir = os.path.join(TestPatch_dir, TestFullImage)
            ARCNN_SRCNN(args, required_width, required_height, output1_MeanPSNR,
                        output2_MeanPSNR, output3_MeanPSNR, MeanTime,
                        PatchName_List, PSNR1_List, PSNR2_List, PSNR3_List,
                        TimeConsume_List)
    dataList = list()
    dataList.extend([PatchName_List, PSNR1_List, PSNR2_List, PSNR3_List,TimeConsume_List])
    dataArray = np.array(dataList)
    FileName = os.path.join(os.getcwd()+'dataArray.csv')
    print(FileName)
    print(dataArray)
    np.savetxt(FileName, dataArray, fmt='%s', delimiter=',')


    print('output1_AvgPSNR: {:.2f}'.format(output1_MeanPSNR.avg))
    print('output2_AvgPSNR: {:.2f}'.format(output2_MeanPSNR.avg))
    print('output3_AvgPSNR: {:.2f}'.format(output3_MeanPSNR.avg))
    print(('MeanTime: {:.2f}'.format(MeanTime.avg)))
    print(PSNR1_List)







