import os
import sys
import argparse
import numpy as np
import torch
from torch.optim import Adam


from network_arch import DDN
from operators import SR
from data import CVDB_Y
from utils import set_gpu, get_timestamp


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
# --------------------------------------------
Training code (GPU) of Deep Decomposition Network (DDN) for Super-resolution (SR) in the paper
@inproceedings{chen2020decomposition,
	author = {Chen, Dongdong and Davies, Mike E},
	title = {Deep Decomposition Learning for Inverse Imaging Problems},
	booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	year = {2020}
}
# --------------------------------------------
Note: The data in the MR Fingerprinting (MRF) examples was from a partner company and we are restricted from sharing. 
      Please refer the following link for the operators used in MRF: https://github.com/echendongdong/PGD-Net
      Users need to specify their own dataset. 
      Our code can be flexibly transferred or directly used on other specific inverse problems.
# --------------------------------------------
"""


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    check_paths(args)
    dtype = set_gpu(args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # the dataset 'BSDS300' can be downloaded from the below link:
    # https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    train_loader = CVDB_Y('BSDS300', batch_size=3, shuffle=True, crop_size=args.img_shape)

    sr = SR(scale_factor=args.scale_factor).cuda()
    sr.eval()
    H, HT = sr.Forward, sr.Backproj
    Pr = lambda x: HT(H(x))
    Pn = lambda x: x - Pr(x)

    ddn = DDN(in_channels=1, out_channels=1, operator=sr, F='dncnn', G='unet', connection_type='cascade').type(dtype)

    optimizer = Adam([{'params': ddn.G.parameters(), 'lr': args.lr, 'weight_decay': args.reg_weight['G']},
                      {'params': ddn.F.parameters(), 'lr': args.lr}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2], gamma=0.1)
    mse_loss = torch.nn.MSELoss().cuda()


    loss_epoch = 0
    print('start training...')
    for e in range(args.epochs):
        ddn.train()
        loss_seq = []
        for x, m, y in train_loader:
            x, m, y = x.type(dtype), m.type(dtype), y.type(dtype)
            # init noise
            n = torch.from_numpy((np.random.normal(0, args.noise_sigam, y.shape))).type(dtype)  # Add Gaussian noise without clipping
            # calculate the psudo-inverse backprojected reconstruction HTy
            HTy = HT(y + n).type(dtype)
            # DDN reconstruction
            x_hat, F, f, g = ddn(HTy, Pr, Pn)
            # calculate the loss
            loss  = mse_loss(x_hat, x) + args.reg_weight['F'] * mse_loss(H(F), n)
            # update parameters (gradient descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_seq.append(loss.item())

        scheduler.step()
        loss_epoch = np.mean(loss_seq)

        print("==>Epoch {}\tloss_total: {:.6f}".format(e + 1, loss_epoch))

        if args.checkpoint_model_dir is not None and (e + 1) % args.checkpoint_interval == 0:
            ddn.eval()
            ckpt = {
                'epoch': e + 1,
                'total_loss': loss_epoch,
                'net_state_dict': ddn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.checkpoint_model_dir, 'ckp_epoch_{}.pt'.format(e)))
            ddn.train()

    # save model
    ddn.eval()
    ckpt = {
        'epoch': args.epochs,
        'total_loss': loss_epoch,
        'net_state_dict': ddn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    save_model_path = os.path.join(args.save_model_dir, args.filename + '.pt')
    torch.save(ckpt, save_model_path)
    print("\nTraining is Done.\tTrained model saved at {}".format(save_model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # set up GPU
    args.cuda = 0
    args.seed = 5213

    # set up super resolution (SR) inverse problem
    args.scale_factor = 2
    args.noise_sigam = 0.1
    args.img_shape = (160, 160) # crop for training

    # set up training hyper-parameters
    args.lr = 1e-3
    args.epochs = 400
    args.batch_size = 4
    # regularization weights for phi_1, phi_2
    args.reg_weight = {'F': 1e-6,
                       'G': 1e-8}
    args.checkpoint_interval = 50

    # set up paths
    args.filename = 'ddn_sr'
    args.prefix = get_timestamp()
    args.save_model_dir = os.path.join('models', args.prefix)
    args.checkpoint_model_dir = os.path.join('models', args.prefix, 'ckp')

    train(args)