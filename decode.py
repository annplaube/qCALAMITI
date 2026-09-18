import os
from glob import glob

in_dir = '/path/to/encoded_volumes'
out_dir = '/path/to/decoded_volumes'  # <-- results path
modality_names = ['R1','R2s', 'PD', 'MTSat']
in_thetas = [
    '/path/to/target_R1_theta.txt',
    '/path/to/target_R2s_theta.txt',
    '/path/to/target_PD_theta.txt',
    '/path/to/target_MTSat_theta.txt',
]
model_path = '/path/to/model_file.pt'

for modality, theta_file in zip(modality_names, in_thetas):
    imgs = os.path.join(in_dir, f'*{modality}*_ori.nii')
    imgs = sorted(glob(imgs))
    num_imgs = len(imgs)

    for img_id, img in enumerate(imgs):
        prefix = os.path.basename(img)
        prefix = prefix.replace('_ori.nii', '')
        print(f'Processing: {prefix}...')
        
        cmd = f'python code/decode_3d.py ' + \
            f'--in-beta {os.path.join(in_dir, prefix)}_beta_axial.nii '+ \
            f'{os.path.join(in_dir, prefix)}_beta_coronal.nii ' + \
            f'{os.path.join(in_dir, prefix)}_beta_sagittal.nii ' + \
            f'--in-theta {theta_file} ' + \
            f'--out-dir {out_dir} ' + \
            f'--prefix {prefix} ' + \
            f'--pretrained-model {model_path} ' + \
            f'--gpu 0 ' + \
            f'--mod {modality} ' + \
            f'--beta-dim 4 ' + \
            f'--theta-dim 2'
        os.system(cmd)
