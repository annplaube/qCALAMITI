import os
from glob import glob

in_dir = '/path/to/image_volumes'
out_dir = '/path/to/encoded_volumes'  # <-- results path
modality_names = ['R1','R2s', 'PD', 'MTSat']
model_path = '/path/to/model_file.pt'

skip_existing = False  # set to True to skip already encoded volumes



for modality in modality_names:
    imgs = glob(f"{in_dir}/*_{modality}_*.nii")
    num_imgs = len(imgs)
    for img_id, img in enumerate(imgs):
        prefix = os.path.basename(img)
        prefix = prefix.replace(".nii", "")
        if os.path.exists(f'{out_dir}/{prefix}_theta.txt') and skip_existing:
            continue
        print(f'{str(img_id+1)}/{str(num_imgs)} Processing: {prefix}')

        cmd = 'python code/encode_3d.py ' + \
                f'--in-img {img} ' + \
                f'--out-dir {out_dir} ' + \
                f'--pretrained-model {model_path} ' + \
                f'--prefix {prefix} ' + \
                f'--mod {modality} ' + \
                f'--avg-theta ' + \
                f'--gpu 0 ' + \
                f'--beta-dim 4 ' + \
                f'--theta-dim 2'
        os.system(cmd)
