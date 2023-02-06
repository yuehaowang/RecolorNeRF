# RecolorNeRF

PyTorch implementation of paper *"RecolorNeRF: Layer Decomposed Radiance Fields for Efficient Color Editing of 3D Scenes"* by Bingchen Gong, Yuehao Wang, Xiaoguang Han, and Qi Dou.

A novel user-friendly color editing approach for neural radiance fields.

**[[Paper]](https://arxiv.org/abs/2301.07958)** **[[Project Website]](https://sites.google.com/view/recolornerf)** **[[Selected Results]](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155168053_link_cuhk_edu_hk/EksrNtKFtGVHssiSxZyuB6sBMFJTE4uTTPJM4lSLLoXcGA?e=wMEAjS)**



https://user-images.githubusercontent.com/6317569/216793671-18ca0551-c668-4bb6-b929-2ba7b5059252.mp4



## Installation

**Tested on Ubuntu 18.04 with PyTorch 1.12.1 and CUDA 11.1.**

Type the commands below to set up the running environment.

```bash
conda create -n recolornerf python=3.8
conda activate recolornerf
# PyTorch (may need to adapt to your environment)
pip install torch torchvision
# PyTorch3D
conda install pytorch3d -c pytorch3d
# Essentials
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia Pillow lpips tensorboard trimesh
conda install -c conda-forge einops
# Palette extraction
conda install -c conda-forge scipy
conda install -c conda-forge cython
CVXOPT_BUILD_GLPK=1 pip install cvxopt
```


## Data

### Support Datasets

* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing](https://drive.google.com/drive/folders/1M-_Fdn4ajDa0CS8-iqejv0fQQeuonpKF)

### Palette Initialization

Download [our customized initial palettes](https://drive.google.com/drive/folders/1XfRea5QnjBr0qjXk6vWurjJt4EI9FWSe?usp=share_link) to `data_palette/` for reproducing our layer decomposition results.

For scenes other than the provided ones, you can use the jupyter notebook `tools/get_palette.ipynb` to generate and customize new palettes.


## Training

Type the command below to train a RecolorNeRF model:

```bash
python run_recolornerf.py --config configs/chair.txt
```

We provide our configurations for 15 scenes in the `configs/` directory. Remember to change the `datadir` option to your dataset path. For more options, please refer to the `utils/opt.py` file.


## Recoloring

We create a jupyter notebook `tools/color_edit.ipynb` for recoloring an optimized RecolorNeRF in a quasi-interactive way. In this jupyter notebook, some simple GUI widgets (like color pickers) and visualization & rendering scripts are provided.


## Citation

If you find our code or paper is helpful, please consider citing:

```
@article{gong2023recolornerf,
  title={RecolorNeRF: Layer Decomposed Radiance Fields for Efficient Color Editing of 3D Scenes},
  author={Gong, Bingchen and Wang, Yuehao and Han, Xiaoguang and Dou, Qi},
  journal={arXiv preprint arXiv:2301.07958},
  year={2023}
}
```


## Acknowledgement

- Our code is based on [TensoRF](https://github.com/apchenstu/TensoRF) and [nerf-pytorch](https://github.com/peihaowang/nerf-pytorch)  (re-implemented by [@peihaowang](https://github.com/peihaowang)).
- Some of the palette extraction code is borrowed from [this repo](https://github.com/JianchaoTan/fastLayerDecomposition).
