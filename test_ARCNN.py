
import time
import scipy.io

import model
from utils import *

def tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)

def tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)

# the image format must be 'jpg'
def denoise(args, im, output_path): # im is the image with artifacts, numpy array
    # Load .mat weights
    weights_dir = os.getcwd()+'\\ARCNN_weights\\'
    weights_path = os.path.join(weights_dir, 'q%d.mat'%args.JPEG_factor)
    if not weights_path:
        print(weights_path+' not exist')
    weights = scipy.io.loadmat(weights_path)
    for k, v in weights.items():
        if '__' not in k:  # Unwanted mega attributes start with __
            a = 1
            # print(k, v.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.ARCNN(weights).to(device).eval()  # eval() ???

    # convert the image to Tensor
    ycbcr = bgr2ycbcr(im.astype(np.float32) / 255).transpose(2, 0, 1)
    start_time = time.time()
    with torch.no_grad():
        im = torch.from_numpy(ycbcr)
        im2 = torch.unsqueeze(im, 0)
        result = net(im2[:, 0:1, :, :])  # result [1,1,512,512]
        comb_result = torch.cat((result, im2[:, 1:3, :, :]),
                                1)  # concatnate two tensor by dimension = 1 # comb_result is in Y'CrCb mode
        for i in range(result.shape[0]):  # why no imwrite
            cv2.imwrite(output_path,
                        (ycbcr2bgr(
                            tensor_to_np(comb_result[i])
                        ) * 255 + 0.5).astype(np.int32))  # Y'CrCb to RGB
        im_denoise = cv2.imread(output_path, -1) # im_denoise shape: (170, 170, 3)
    return im_denoise

    #
    # with torch.no_grad(): #im in numpy, I makes some wrong
    #     im = torch.from_numpy(ycbcr) # TODO: dataset 的问题，之前的dataloader 的 dataset struct 是：{image_name, ycbcr}。Dataload之后，变成torch形式 net输入是4维的，太神奇
    #     im = im.to(device) # im in YCrCb channels
    #     result = net(im[:,0:1,:,:]) # We only take the Y channel, and get the Y' = net(Y)
    #     comb_result = torch.cat((result, im[:,1:3,:,:]), 1) # concatnate two tensor by dimension = 1 # comb_result is in Y'CrCb mode
    #     for i in range(result.shape[0]): # result.shape[0] = 3
    #         cv2.imwrite(output_path,
    #             (ycbcr2bgr(
    #                 tensor_to_np(comb_result[i])
    #             )*255+0.5).astype(np.int32)) # Y'CrCb to RGB
    #     im_denoise = cv2.imread(output_path, -1)
    # return im_denoise


