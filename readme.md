## Introduction
This repository is a TensorFlow version of [You said that?](https://arxiv.org/abs/1705.02966). Demos are shown in [videos](./videos). 

Since [Matlab code](https://github.com/joonson/yousaidthat) is given by the author. The basic idea of my work is converting the Matlab model to TensorFlow. Origin Matlab model parameters are stored in the intermediate .mat file [here](https://drive.google.com/drive/folders/191J8_ZMoj7Um3C9t2MqDMfV2bxlMdh9o?usp=sharing). TensorFlow variables are initialized using these files. 

Note1: 5 faces are used in the Matlab code. Only 1 face is used as the identity. However, both intermediate .mat files are given. `v201_param_cell_idnum1.mat` means only 1 face used as identity.

Note2: The deblur model is also implemented, and the intermediate .mat file is also given, which is named `v114_param_cell.mat`.

Note3: the upsample layer is defined by the author in the given Matlab code. So the upsample layer is not exact same with Matlab model. Both bilinear upsample and deconvolution are tried in the TensorFlow version. And the deconvolution performs better. 

The code is in a mess. Some code is deprecated. Sorry that I don't have time to sort them out. Different models are indicated by its file name. For example, `_halfface.py` means L1 loss is only applied in the lower half face. `_save_step0.py` means not a single step is trained. The model is only initialized by Matlab parameters. It is used to verify that the Matlab parameters really worked in TensorFlow. Besides, GAN is implemented. The AutoEncoder is used as G, and the D comes from [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py).

If you have any questions, please contact me.