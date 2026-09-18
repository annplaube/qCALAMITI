#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import torch
from modules.model import qCALAMITI
from modules.dataset import obtain_single_img, INTENSITY_SCALE

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='DisentangledVAE')
    parser.add_argument('--in-img', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='sub1')
    parser.add_argument('--mod', type=str, required=True)
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--avg-theta', default=False, action='store_true')
    parser.add_argument('--theta-only', default=False, action='store_true')
    parser.add_argument('--mask', type=str, default=None)
    args = parser.parse_args(args)

    # initialize model
    encoder = qCALAMITI(beta_dim = args.beta_dim,
                       theta_dim = args.theta_dim,
                       train_sample = 'st_gumbel_softmax',
                       valid_sample = 'argmax',
                       pretrained_model = args.pretrained_model,
                       gpu = args.gpu)
    # TODO: remove permutation in obtain_single_img once trained with correct loading
    img_vol, img_slc, img_hdr, img_affine = obtain_single_img(
        img_dir=args.in_img,
        avg_theta=args.avg_theta,
        minmax=INTENSITY_SCALE[args.mod],
    )
    img_vol = img_vol.float()

    if args.mask is not None:
        mask_vol, mask_slc,_,_ = obtain_single_img(
            img_dir=args.mask,
            avg_theta=args.avg_theta,
            minmax=(0,1),
        )
        img_vol = img_vol * mask_vol
        img_slc = list((torch.stack(img_slc) * torch.stack(mask_slc)).unbind(0))
    
    # axial
    encoder.encode_single_vol(
        img_vol,
        img_slc,
        args.out_dir,
        args.prefix,
        orientation='axial',
        img_hdr=img_hdr,
        img_affine=img_affine,
        theta_only=args.theta_only
    )
    # coronal
    encoder.encode_single_vol(
        img_vol,
        img_slc,
        args.out_dir,
        args.prefix,
        orientation='coronal',
        img_hdr=img_hdr,
        img_affine=img_affine,
        theta_only=args.theta_only
    )
    # sagittal
    encoder.encode_single_vol(
        img_vol,
        img_slc,
        args.out_dir,
        args.prefix,
        orientation='sagittal',
        img_hdr=img_hdr,
        img_affine=img_affine,
        theta_only=args.theta_only
    )
if __name__ == '__main__':
    main()
