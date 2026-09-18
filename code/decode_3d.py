#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
import nibabel as nib
import matplotlib.pyplot as plt
import os
   
from modules.model import qCALAMITI
from modules.dataset import INTENSITY_SCALE, load_beta, load_theta

def recover_scale(img, vmin, vmax):
    return img * (vmax - vmin) + vmin

def combine_imgs(img_dirs, out_dir, prefix, modality=None):
    # obtain images
    imgs = []
    for img_dir in img_dirs:
        img_file = nib.load(img_dir)
        img_vol = img_file.get_fdata().astype(np.float32)
        if modality is not None:
            img_vol = recover_scale(img_vol, *INTENSITY_SCALE[modality])
        img = torch.from_numpy(img_vol)
        img_hdr = img_file.header
        img_affine = img_file.affine
        imgs.append(img.numpy())

    # calculate median
    img_cat = np.stack(imgs, axis=-1) 
    img_median = np.median(img_cat, axis=-1)
    img_save = nib.Nifti1Image(img_median, img_affine, img_hdr)
    file_name = os.path.join(out_dir, f'{prefix}_med_fusion.nii.gz')
    nib.save(img_save, file_name)

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='DisentangledVAE')
    parser.add_argument('--in-beta', type=str, nargs='+', required=True)
    parser.add_argument('--in-theta', type=str,  required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='sub1')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--mod', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args(args)

    # initialize model
    decoder = qCALAMITI(
        beta_dim = args.beta_dim,
        theta_dim = args.theta_dim,
        train_sample = 'st_gumbel_softmax',
        valid_sample = 'argmax',
        pretrained_model = args.pretrained_model,
        gpu = args.gpu
    )
    # load data
    theta = load_theta(theta_dir = args.in_theta)
    orientations = ['axial', 'coronal', 'sagittal']
    for beta_dir, orientation in zip(args.in_beta, orientations):
        beta, img_hdr, img_affine = load_beta(beta_dir)
        decoder.decode_single_vol(
            beta,
            theta,
            args.out_dir,
            args.prefix,
            orientation=orientation,
            img_hdr=img_hdr,
            img_affine=img_affine
        )

    # fusion
    decode_img_dirs = []
    for orientation in orientations:
        decode_img_dirs.append(os.path.join(args.out_dir, f'{args.prefix}_recon_{orientation}.nii'))
    combine_imgs(decode_img_dirs, args.out_dir, args.prefix, args.mod)

if __name__ == '__main__':
    main()

