# qCALAMITI

Image-based harmonization approach for Multi-Parametric Mapping (MPM) of the brain.

See publication
> Laube A, Leutritz T, Weiskopf N, et al. qCALAMITI – unsupervised image harmonization of multi-parameter
quantitative MRI. Submitted.

Based on
> Zuo L, Dewey BE, Liu Y, et al. Unsupervised MR harmonization by learning disentangled representations using information bottleneck theory. NeuroImage. 2021;243. doi:10.1016/j.neuroimage.2021.118569
https://doi.org/10.1016/j.neuroimage.2021.118569

Original code: https://iacl.ece.jhu.edu/index.php?title=CALAMITI

### How to use qCALAMITI

qCALAMITI has been developed for MPM from the hMRI toolbox [Tabelow et al., 2019](10.1016/j.neuroimage.2019.01.029). Before attempting harmonization, one needs to decide which domains (scanners, software versions, clinical centers) count as "different", and which ones as the "same". The image volumes should be kept in different folders.

Image volumes are expected to be centered at the anterior commissure and in standard MNI orientation.

#### Training
Training requires access to a GPU.


```bash
$ python qCALAMITI/code/train_harmonization.py \
    --dataset-dirs /path/to/SiteA /path/to/SiteB /path/to/SiteC \
    --data-names R1 R2s PD MTSat \
    --orientation AXIAL CORONAL SAGITTAL \
    --epochs 100 \
    --gpu 0 \
    --batch-size 4 \
    --out-dir /path/to/results/dir \
    --beta-dim 4 \
    --theta-dim 2
```
Tensorboard logs and trained models are written to the results directory.

#### Inference
Once trained, encoding and decoding can be done with the scripts provided, simply add the parameters and relevant paths in the files.

```bash
$ python qCALAMITI/encode.py
$ python qCALAMITI/decode.py
```

Decoding requires target &theta; values, which most likely will need to be obtained by inspecting the distributions of &theta; in the datasets.

The target &theta; values for each contrast must be saved as comma-separated values in separate files.

example-theta-R1.csv
```{r, attr.source='.numberLines'}
10.0,6.5
```
