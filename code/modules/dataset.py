import os
from glob import glob
import nibabel as nib
import torch
import numpy as np
import random
from torchio.transforms import CropOrPad
from torchvision.transforms import Compose, Pad, CenterCrop, ToTensor, Resize, ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data.dataset import Dataset
from PIL import Image

INTENSITY_SCALE = {"PD":(0,200), "MTSat": (0,15), "R1": (0,2), "R2s":(0,100)}


class QuantitativeToTensor:
    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, x):
        is_tensor = torch.is_tensor(x)

        # --- convert to tensor if needed ---
        if not is_tensor:
            x = torch.from_numpy(x)
        else:
            # keep original device
            device = x.device
        x = x.float()
        x = (x - self.vmin) / (self.vmax - self.vmin)
        x = torch.clamp(x, 0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)
        x = x.float() / 255.0
        if is_tensor:
            x = x.to(device)
        return x


DEFAULT_TRANSFORM = Compose([Pad(60), CenterCrop((288,288))])


def load_img_from_file(img_path, minmax):
    """returns image"""
    img = nib.load(img_path).get_fdata().astype(np.float32).squeeze()
    transform = Compose([QuantitativeToTensor(*minmax), DEFAULT_TRANSFORM])
    img = transform(img)
    return torch.from_numpy()(np.array(img))


def load_vol_from_file(img_path, minmax):
    img_file = nib.load(img_path)
    img_vol = torch.from_numpy(img_file.get_fdata().astype(np.float32).squeeze())
    transform = Compose([QuantitativeToTensor(*minmax), CropOrPad(288)])
    img_vol = transform(img_vol.unsqueeze(0)).squeeze()
    return img_vol


def obtain_single_img(img_dir, avg_theta, minmax=(0,255)):
    """
    img_dir: path to image volume
    avg_theta: (BOOL) take average theta from 20 middle slices (True) or theta from single middle slice (False)
    minmax: (tuple) define normalisation for QuantitativeToTensor
    """
    img_file = nib.load(img_dir)
    img_vol = load_vol_from_file(img_dir, minmax)

    img_slcs = []
    if avg_theta:
        for slc in range(134,154):
            img_slc = img_vol[slc].unsqueeze(0)               # torch.Size([1,288,288])
            img_slc = img_slc.repeat(img_vol.shape[0],1,1,1)  # torch.Size([288, 1, 288, 288])
            img_slcs.append(img_slc)                          # list
    else:
        img_slc = img_vol[144].unsqueeze(0)
        img_slc = img_slc.repeat(img_vol.shape[0],1,1,1)    # single torchSize([288,1,288,288])
        img_slcs.append(img_slc)                            # list with one entry (to have same size as above)
    return img_vol, img_slcs, img_file.header, img_file.affine


def load_beta(beta_dir):
    """return beta volume with dim [x, ch, y, z]"""
    beta_file = nib.load(beta_dir)
    beta = torch.from_numpy(np.array(beta_file.get_fdata().astype(np.float32))).permute(0,3,1,2)
    img_hdr = beta_file.header
    img_affine = beta_file.affine
    return beta, img_hdr, img_affine


def load_theta(theta_dir):
    try:
        theta = torch.FloatTensor([float(value) for value in theta_dir])
    except ValueError:
        theta = torch.FloatTensor(np.loadtxt(theta_dir, delimiter=',')).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return theta


class MultiOrientationImages(Dataset):
    """
    To train 3D fusion network
    INPUT
    * dataset_dir: format: '[dataset_dir]/train' and  '[dataset_dir]/valid'
    * data_name: 'T1' or 'T2'
    * Note: file-prefix should be '*_ori.nii*' for original images and '*[axial/coronal/sagittal]_recon.nii*' for self-recon images 
    """
    def __init__(self, dataset_dir, data_name, mode='train'):
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.data_name = data_name
        self.ori_imgs = self._get_files()

    def _get_files(self):
        ori_imgs = os.path.join(self.dataset_dir, self.mode, f'*{self.data_name}_ori.nii*')
        ori_imgs = sorted(glob(ori_imgs))
        return ori_imgs

    def __len__(self):
        return len(self.ori_imgs)

    def __getitem__(self, idx:int):
        imgs = []
        orientations = ['axial', 'coronal', 'sagittal']
        ori_img = self.ori_imgs[idx]
        for orientation in orientations:
            str_id = ori_img.find('_ori.nii*')
            img_path = sorted(glob(f'{ori_img[:str_id]}_*_{orientation}_recon.nii*'))[0]
            img = np.array(nib.load(img_path).get_fdata().astype(np.float32))
            img = ToTensor()(img)
            imgs.append(img.float().permute(2,1,0).permute(2,0,1).unsqueeze(0))
        ori_img_file = nib.load(ori_img)
        ori_img = np.array(ori_img_file.get_fdata().astype(np.float32))
        ori_img = ToTensor()(ori_img)
        return imgs, ori_img.float().permute(2,1,0).permute(2,0,1).unsqueeze(0)  


class PairedMRI(Dataset):
    def __init__(self, dataset_dirs, data_names, orientations, mode="train", intensity_scale=INTENSITY_SCALE):
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.data_names = data_names
        self.orientations = orientations
        self.intensity_scale = intensity_scale
        self.imgs, self.dataset_ids = self._get_files()

    def _get_files(self):
        imgs = []
        dataset_ids = []
        for data_name in self.data_names:
            img_list = []
            dataset_id_list = []
            for dataset_id, dataset_dir in enumerate(self.dataset_dirs):
                vol_path = os.path.join(dataset_dir, self.mode, f'*{data_name}*.nii*')
                for img_path in sorted(glob(vol_path)):
                    for orientation in self.orientations:
                        axis = {"AXIAL": 0, "CORONAL": 1, "SAGITTAL": 2}[orientation]
                        num_slices = {"AXIAL": 80, "CORONAL": 100, "SAGITTAL": 100}[orientation]
                        vol = load_vol_from_file(img_path, self.intensity_scale[data_name])
                        slices = self.get_central_slices(vol, axis=axis, num_slices=num_slices, eps=0)
                        for i in slices:
                            img_list.append((img_path, axis, i))
                            dataset_id_list.append(dataset_id if data_name=="R1" else -1.0*dataset_id)
            imgs.append(img_list)
            dataset_ids.append(dataset_id_list)
        return tuple(imgs), dataset_ids

    def __len__(self):
        return len(self.imgs[0])

    def __getitem__(self, idx:int):
        imgs = []
        dataset_ids = []
        other_imgs = []
        for modality_id in range(len(self.imgs)):
            data_name = self.data_names[modality_id]
            minmax = self.intensity_scale[data_name]
            img_path, img_axis, img_slice = self.imgs[modality_id][idx]
            img_vol = load_vol_from_file(img_path, minmax)
            img = self.get_slice(img_vol, img_axis, img_slice)
            dataset_id = self.dataset_ids[modality_id][idx]
            # random other choice in same subject:
            random_slice = random.choice(self.get_central_slices(img_vol, axis=0, num_slices=80, eps=0))
            other_img = self.get_slice(img_vol, 0, random_slice)
            imgs.append(img)
            dataset_ids.append(dataset_id)
            other_imgs.append(other_img)

        # make sure both T1 and T2 have the same FOV
        img0 = imgs[0]
        img1 = imgs[1]
        msk0 = img0.ge(1e-3)
        msk1 = img1.ge(1e-3)
        msk = msk0 & msk1
        imgs[0][~msk] = 0
        imgs[1][~msk] = 0
        return tuple(imgs), dataset_ids, tuple(other_imgs)

    def get_central_slices(self, vol, axis=0, num_slices=100, eps=0):
        dims = list(range(vol.ndim))
        reduce_dims = [d for d in dims if d != axis]

        mask = (vol.abs().sum(dim=reduce_dims) > eps)
        indices = torch.where(mask)[0]
        if len(indices) == 0:
            return indices

        center = (indices[0] + indices[-1]) // 2
        half = num_slices // 2

        start = int(max(center - half, indices[0]))
        end   = int(min(center + half, indices[-1] + 1))
        return torch.arange(start, end, device=vol.device)

    def get_slice(self, img_vol, axis, slice_idx):
        slices = [slice(None)] * img_vol.ndim
        slices[axis] = slice_idx
        return img_vol[tuple(slices)].unsqueeze(0)
