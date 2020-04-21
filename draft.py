"""
prepare the test input for baseline1
each input is an image-file, whose size is 256*256.
this code would generate the test inputs.
the output of this code will be saved in 2 files. One is the hr image(.bmp), another is the lr image(jpg).
"""
import argparse
import glob
import h5py
import numpy as np
from PIL import Image, ImageFilter
from utils import convert_rgb_to_y, mkdir
import os
import cv2

def eval(args):
    eval_list = os.listdir(args.images_dir)
    count = 0
    for num, img_name in enumerate(eval_list):
        print(img_name)
        dir_bmp = os.path.join(os.getcwd(),'/database/test/'+img_name)
        img_jpgname = img_name.replace('.bmp','.jpg')
        dir_jpg = os.path.join(os.getcwd(), '/database/test/'+img_jpgname)
        mkdir(dir_bmp), mkdir(dir_jpg)
        image_path = os.path.join(args.images_dir, img_name)
        image_jpgpath = os.path.join(args.jpg_image_dir, img_jpgname)
        hr = Image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC) # hr.size = 512 -> 510
        hr_blur = hr.filter(ImageFilter.GaussianBlur(2))
        lr = hr_blur.resize((hr_width // args.scale, hr_height // args.scale), resample=Image.BICUBIC) # lr.size = 510/3 -> 170
        lr.save(image_jpgpath, quality=args.JPEG_factor)
        img_pil_jpg = Image.open(image_jpgpath).convert('RGB')
        # TODO: add the blur, downsample and denoise.
        lr = img_pil_jpg.resize((lr.width * args.scale, lr.height * args.scale), resample=Image.BICUBIC) # lr'.size = 170 -> 170*3=510
        # TODO: If using the sub-pixel, no need to get the lr'. Because after getting through the network, the image size will be enlarged automatically.
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        # print(np.shape(lr))
        # cutting the pairs of patches
        patch_index = 0
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patch_name = img_name.replace('.bmp', '_{}.jpg'.format(str(patch_index)))
                hr_patch_name = img_name.replace('.bmp', '_{}.bmp'.format(str(patch_index)))
                print(hr_patch_name)
                lr_patch = lr[i:i + args.patch_size, j:j + args.patch_size, :]
                hr_patch = hr[i:i + args.patch_size, j:j + args.patch_size, :]
                lr_patch = cv2.cvtColor(lr_patch, cv2.COLOR_BGR2RGB)
                hr_patch = cv2.cvtColor(hr_patch, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(dir_jpg, lr_patch_name), lr_patch)
                cv2.imwrite(os.path.join(dir_bmp, hr_patch_name), hr_patch)
                count = count+1
                patch_index = patch_index+1
    print(count)

# input the dir of the ground true 512*512 bmp image with JPEG-factor and SR scale
# output the images that are blurred, downsampled and denoised, stored in another jpg_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default=r'E:\PythonCode\bishe\Database_baseline2\RAISE_test_small')
    parser.add_argument('--jpg-image-dir', type=str, default=(os.path.join(os.getcwd(), 'database'))) ### change the database
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--function', type=str, default='test') # to make the test h5 file, choose eval function
    parser.add_argument('--JPEG-factor', type=int, default=40)
    args = parser.parse_args()
    args.jpg_image_dir = os.path.join(args.jpg_image_dir, args.function)
    args.jpg_image_dir = os.path.join(args.jpg_image_dir, ('jpg_FullImage_'+args.function))
    mkdir(args.images_dir), mkdir(args.jpg_image_dir)
    action = args.function
    print(args)

    if action == 'train':
        # train(args)
        print('train')
    elif action == 'eval':
        eval(args)
        print('eval')
    elif action == 'test':
        eval(args)
        print('test')
    else:
        print('please enter train, eval or test in --function')


