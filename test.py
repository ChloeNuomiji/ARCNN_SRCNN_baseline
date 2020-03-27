import cv2
import os
import numpy as np
import torch
import model
import scipy.io
from torch.utils.data.dataset import Dataset

def bgr2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2bgr(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2BGR)
    return im_rgb

def tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)

class YcbCrLoader(Dataset): # ?? # combine the ycbcr_image and file_name
    def __init__(self, root): # what to input in the file
        self.root = root
        self.imgs = os.listdir(root)

    def __getitem__(self, idx):
        im = cv2.imread(os.path.join(self.root, self.imgs[idx]))
        ycbcr = bgr2ycbcr(im.astype(np.float32)/255).transpose(2, 0, 1)
        #print(idx)
        return self.imgs[idx], ycbcr

    def __len__(self):
        return len(self.imgs)

# YcbCrLoader input the dir of image
dir = os.path.join(os.getcwd(),'test_LRLQ')
image_name = 'r40d35a09t.jpg'

# Load .mat weights
weights_dir = os.getcwd() + '\\ARCNN_weights\\'
weights_path = os.path.join(weights_dir, 'q40.mat' )
if not weights_path:
    print(weights_path + ' not exist')
weights = scipy.io.loadmat(weights_path)
for k, v in weights.items():
    if '__' not in k:  # Unwanted mega attributes start with __
        a = 1
        # print(k, v.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = model.ARCNN(weights).to(device).eval()  # eval() ???



# 一、 load dataset
dataset = YcbCrLoader(dir)
# print(dataset.__getitem__(0))
loader = torch.utils.data.DataLoader(dataset, batch_size=1) #

# 二、compare to confirm if you extract the Y channel
print('load single image')
image_file = os.path.join(dir, image_name)
output_path = os.path.join(os.getcwd(), 'r40d35a09t_arcnn.jpg')
image = cv2.imread(image_file)
ycbcr = bgr2ycbcr(image.astype(np.float32) / 255).transpose(2, 0, 1) # shape = (3,512,512)
with torch.no_grad():
    im = torch.from_numpy(ycbcr)
    im2 = torch.unsqueeze(im, 0)
    im_input = im2[:, 0:1, :, :]
    result = net(im2[:,0:1, :, :])  #
    print('result size: ', result.size())
    comb_result = torch.cat((result, im2[:, 1:3, :, :]),1)  # concatnate two tensor by dimension = 1 # comb_result is in Y'CrCb mode
    for i in range(result.shape[0]):  # it can!!!! imwrite
        cv2.imwrite(output_path, (ycbcr2bgr(tensor_to_np(comb_result[i])) * 255 + 0.5).astype(np.int32))  # Y'CrCb to RGB
    #im_denoise = cv2.imread(output_path, -1)  # im_denoise shape: (170, 170, 3)
    #print(np.shape(im_denoise))
# print(name)

print('load dataset')
count = 0
for name, im_4d in loader:
    count += 1
    print(count)
    print(im_4d[:,0:1, :, :])
    im_3d = torch.squeeze(im) # im2 in shape [3,512,512]
    #result = net(im2[0:1, :, :])  # We only take the Y channel, and get the Y' = net(Y)
    #comb_result = torch.cat((result, im[:, 1:3, :, :]),1)  # concatnate two tensor by dimension = 1 # comb_result is in Y'CrCb mode
    #print(name)
    raise Exception('STOP')


