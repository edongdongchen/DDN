import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import imresize

"""
PyTorch implementation of forward/adjoint operators for supre-resolution (SR) inverse problem in the paper
@inproceedings{chen2020decomposition,
	author = {Chen, Dongdong and Davies, Mike E},
	title = {Deep Decomposition Learning for Inverse Imaging Problems},
	booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2020}
}
"""

class SR(nn.Module):
    def __init__(self, scale_factor=2, dtype=torch.cuda.FloatTensor):
        super(SR, self).__init__()
        self.upsample_scale = scale_factor
        self.downsample_scale = 1/scale_factor

        self.dtype = dtype

    def forward(self, x):
        return self.Downsample(x)

    def Downsample(self, x, type=None):
        assert len(x.shape)==4, '4D input: NCHW'
        if type=='matlab':
            print(x.shape)
            y = torch.cat([imresize(x[i], scale=self.downsample_scale)
                          .view(1, x.shape[1], int(self.downsample_scale*x.shape[2]), int(self.downsample_scale*x.shape[3]))
                           for i in range(x.shape[0])], dim=0)# sample by sample
        else:
            #use torch.bicubic as default, no anti-aliasing
            y = F.interpolate(x, scale_factor=self.downsample_scale, mode="bicubic")#bicubic, bilinear
        return y.type(self.dtype)


    def Upsample(self, x):
        y = F.interpolate(x, scale_factor=self.upsample_scale, mode="bicubic")
        return y.type(self.dtype)

    def Forward(self, x):
        return self.Downsample(x)

    def Backproj(self, x):
        return self.Upsample(x)


    def Pr(self, x):
        return self.Backproj(self.Forward(x))

    def Pn(self, x):
        return x - self.Pr(x)