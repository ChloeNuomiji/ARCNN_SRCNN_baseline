"""
test_SRCNN

"""
import argparse
import os
import torch
#import torch.backimg_formats.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

##############
# do not define the test_SRCNN as SRCNN, since it has been used in the model.py
def SRCNN2(args, image_file): # CHANGE TO INPUT THE after-resize IMAGE FILE, SO IN THE OUTPUT3, NEED TO STORE THE denoise+resize image
    # load the SRCNN weights model
    #cudnn.benchmark = True
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    weights_dir = os.getcwd()+'\\SRCNN_outputs\\x{}\\'.format(args.SR_scale) #
    weights_file = os.path.join(weights_dir,'best.pth') ###
    if not weights_file:
        print(weights_file+' not exist')
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval() # model set in evaluation mode

    img_format = image_file[-4:]
    image = pil_image.open(image_file).convert('RGB') # 512

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0) # output2.size 510

    # psnr = calc_psnr(y, preds)
    # print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0) # tensor -> np

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0]) # why transpose
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    return output  ## type in pil_image
    #output.save(args.image_file.replace(img_format, ('_srcnn_x{}'+img_format).format(args.SR_scale)))
