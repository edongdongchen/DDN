import os
from network_arch import DDN
from operators import SR
from data import CVDB
from utils import *

'''
PyCharm (Python 3.6.9)
PyTorch 1.3
Windows 10 or Linux
Dongdong Chen (d.chen@ed.ac.uk)
github: https://github.com/echendongdong/DDN
If you have any question, please feel free to contact with me.
Dongdong Chen (e-mail: d.chen@ed.ac.uk)
by Dongdong Chen (01/March/2020)
'''

"""
Testing code (GPU) of Deep Decomposition Network (DDN) for Super-resolution (SR) in the paper
@inproceedings{chen2020decomposition,
	author = {Chen, Dongdong and Davies, Mike E},
	title = {Deep Decomposition Learning for Inverse Imaging Problems},
	booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2020}
}
"""

# set up super-resolution (SR) inverse problem
scale_factor=2
noise_sigma=0.1
dtype = set_gpu(gpu_id=0)

# init SR opeartor
sr = SR(scale_factor).type(dtype)
sr.eval()
H, HT = sr.Forward, sr.Backproj
Pr = lambda x: HT(H(x))
Pn = lambda x: x - Pr(x)

# load trained DDN
ddn = DDN(in_channels=1, out_channels=1, operator=None, F='dncnn', G='unet', connection_type='cascade').type(dtype)
# the trained model 'DDN-joint-cascade.pt' can be downloaded from
# https://drive.google.com/file/d/1tTAcxAlA3ZIvEKUv5x1Qd9XJy4sLHp24/view
# move the downloaded model into the folder 'models'
ddn.load_state_dict(torch.load(os.path.join('models', 'DDN-joint-cascade.pt')))
ddn.cuda().eval()

# init groundtruth HR
dataloader = CVDB('Set1', batch_size=1, shuffle=False, crop_size=None)
for data in dataloader:
    x = data[0]
    break

YCbCr = torch.einsum('nchw->cnhw', x).type(dtype)
Y, Cb, Cr = [YCbCr[i].unsqueeze(1) for i in range(3)]

# init measurement LR (only Y channel is used)
LR = sr.Downsample(Y)
n = torch.from_numpy(np.random.normal(0, 0.1, LR.shape)).type(dtype)
LR_n = LR + n

# calculate pseudo-inverse reconstruction (back-projection, BP)
BP = sr.Upsample(LR_n)

# calculate Y_ddn: DDN reconstruction of x (Y channel), F: output of F net, f: recovered range-component, g: recovered nullspace-component
Y_ddn, F, f, g = ddn(BP, Pr, Pn)

# plot reconstruction results and psnr results
title = ['HR: $x$', 'Bicubic: $H^{\dag}y_{\epsilon}$', 'DDN', '$H^{\dag}y_{\epsilon} + P_r(F)$', '$P_n(G)$']
img = [torch_ycbcr_to_rgb(x[0]),
       torch_ycbcr_to_rgb(torch.cat((BP, Cb, Cr), dim=1)[0]),
       torch_ycbcr_to_rgb(torch.cat((Y_ddn, Cb, Cr), dim=1)[0]),
       torch_ycbcr_to_rgb(torch.cat((BP - f, Cb, Cr), dim=1)[0]),
       torch_to_np(g)]
psnr = ['', torch_psnr(Y[0], BP[0]), torch_psnr(Y[0], Y_ddn[0]), '', '']

print('title, img, psnr', len(title), len(img), len(psnr))

import matplotlib.pyplot as plt
for i in range(len(img)):
    print(i)
    plt.subplot(1,5,i+1)
    plt.imshow(img[i], cmap='RdGy')
    plt.title(title[i])
    plt.xlabel(psnr[i])
    ax = plt.gca()
    ax.set_xticks([]), ax.set_yticks([])
plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.9, hspace=0.02, wspace=0.02)
plt.show()