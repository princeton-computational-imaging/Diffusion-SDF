# Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions

[**Paper**](https://arxiv.org/abs/2211.13757) | [**Supplement**](https://light.princeton.edu/wp-content/uploads/2023/03/diffusionsdf_supp.pdf) | [**Project Page**](https://light.princeton.edu/publication/diffusion-sdf/) <br>

~~(Update Jan 19th 2023) Repo created. Code will be released early February. Stay tuned!~~ <br>
(Update Feb 12th 2023) Training and generation / visualization code are released. Instructions provided below. <br>
(Update Mar 4th 2023) When refactoring the code we seemed to introduce some bugs to the diffusion model when training with pytorch lightning, so the third training stage does not work as expected. We recommend simply generating after the second stage, and we will update the repo when we identify the issue. 

This repository contains the official implementation of <br> 
**Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions** <br>
[Gene Chou](https://genechou.com), [Yuval Bahat](https://sites.google.com/view/yuval-bahat/home), [Felix Heide](https://www.cs.princeton.edu/~fheide/) <br>


If you find our code or paper useful, please consider citing
```bibtex
@article{chou2022diffusionsdf,
title={DiffusionSDF: Conditional Generative Modeling of Signed Distance Functions},
author={Gene Chou and Yuval Bahat and Felix Heide},
journal={arXiv preprint arXiv:2211.13757},
year={2022}
}
```


<!-- ```cpp
root directory
  ├── config  
  │   └── // folders for checkpoints and training configs
  ├── data  
  │   └── // folders for data (in csv format) and train test splits (json)
  ├── models  
  │   ├── // models and lightning modules; main model is 'combined_model.py'
  │   └── archs
  │       └── // architectures such as PointNets, SDF MLPs, diffusion network..etc
  ├── dataloader  
  │   └── // dataloaders for different stages of training and generation
  ├── utils  
  │   └── // reconstruction and evaluation
  ├── metrics  
  │   └── // reconstruction and evaluation
  ├── diff_utils  
  │   └── // helper functions for diffusion
  ├── environment.yml  // package requirements
  ├── train.py  // script for training, specify the stage of training in the config files
  ├── test.py  // script for testing, specify the stage of testing in the config files
  └── tensorboard_logs  // created when running any training script
  
``` -->

## Installation
We recommend creating an [anaconda](https://www.anaconda.com/) environment using our provided `environment.yml`:

```
conda env create -f environment.yml
conda activate diffusionsdf
```

## Dataset
For training, we preprocess all meshes and store query coordinates and signed distance values in csv files. Each csv file corresponds to one object, and each line represents a coordinate followed by its signed distance value. See `data/acronym` for examples. Modify the dataloader according to your file format. <br>

When sampling query points, make sure to also **sample uniformly within the 3D grid space** (i.e. from (-1,-1,-1) to (1,1,1)) rather than only sampling near the surface to avoid artifacts. For each training batch, we take 70% of query points sampled near the object surface and 30% sampled uniformly in the grid. `grid_source` in our dataloader and config file refers to the latter. <br>

## Training
As mentioned in our [paper](https://arxiv.org/abs/2211.13757), there are three stages of training. All corresponding config files can be found in the `config` folders. Logs are created in a `tensorboard_logs` folder in the root directory. We recommend tuning the `"kld_weight"` when training the joint SDF-VAE model as it enforces the continuity of the latent space. A higher value (e.g. 0.1) will result in better interpolation and generalization but sometimes more artifacts. A lower value (e.g. 0.00001) will result in worse interpolation but higher quality of generations. <br>

1. Training SDF modulations

```
cd train_sdf
python train.py -e config/stage1_sdf/ -b 32 -w 8    # -b for batch size, -w for workers, -r to resume training
```
Training notes: For Acronym / ShapeNet datasets, the loss should go down to $6 \sim 8 \times 10^{-4}$ after a few days. Run testing to visualize whether the quality of reconstructed shapes is sufficient. The quality of reconstructions will carry over to the quality of generations. Note that the dimension of the VAE latent vectors will be 3 times `"latent_dim"` in `"SdfModelSpecs"` listed in the config file.

2. Training the diffusion model using the modulations extracted from the first stage 

```
# extract the modulations / latent vectors, which will be saved in a "modulations" folder in the config directory
# the folder needs to correspond to "data_path" in the diffusion config files
cd train_sdf
python test.py -e config/stage1_sdf/ -b 32 -w 8 -r last

# unconditional
cd train_diffusion
python train.py -e config/stage2_uncond/ -b 32 -w 8 

# conditional
cd train_diffusion
python train.py -e config/stage2_cond/ -b 32 -w 8 
```
Training notes: When extracting modulations, we recommend filtering based on the chamfer distance. See `test_modulations()` in `test.py` for details. `diff100` should approach 0 while `diff1000` can remain relatively high. Some notes on the conditional config file:  `"perturb_pc":"partial"`, `"crop_percent":0.5`, and `"sample_pc_size":128` refers to cropping 50% of a point cloud with 128 points to use as condition. `dim` in `diffusion_model_specs` needs to be the dimension of the latent vector, which is 3 times `"latent_dim"` in `"SdfModelSpecs"`. <br>

<!-- For training curves in tensorboard, `diff100` should approach 0 while `diff1000` can remain relatively high. Apply highest smoothing in tensorboard when viewing. See `models/combined_model.py` for details. -->


3. End-to-end training using the saved models from above (**Currently this training stage has bugs, will update after fixing.**)

```
# unconditional
cd train_sdf
python train.py -e config/stage3_uncond/ -b 32 -w 8 -r finetune     # training from the saved models of first two stages
python train.py -e config/stage3_uncond/ -b 32 -w 8 -r last     # resuming training if third stage has been trained 


# conditional
cd train_sdf
python train.py -e config/stage3_cond/ -b 32 -w 8 -r finetune    # training from the saved models of first two stages
python train.py -e config/stage3_cond/ -b 32 -w 8 -r last     # resuming training if third stage has been trained 
```
Training notes: The config file needs to contain the saved checkpoints for the previous two stages of training. View tensorboard logs to determine when to stop training. The sdf loss (not generated sdf loss) should approach $6 \sim 8 \times 10^{-4}$.

## Testing
1. Testing SDF reconstructions and saving modulations

After the first stage of training, visualize / test reconstructions and save modulations:
```
# extract the modulations / latent vectors, which will be saved in a "modulations" folder in the config directory
# the folder needs to correspond to "data_path" in the diffusion config files
cd train_sdf
python test.py -e config/stage1_sdf/ -r last
```
A `recon` folder in the config directory will contain the `.ply` reconstructions and a `cd.csv` file that logs Chamfer Distance (CD). A `modulation` folder will contain `latent.txt` files for each SDF. The `modulation` folder will be the data path to the second stage of training.

2. Generations 

Meshes can be generated after the second or third stage of training. **Currently third training stage has bugs, will update after fixing.**
```
cd train_sdf
python test.py -e config/stage3_uncond/ -r finetune  # generation after second stage 
python test.py -e config/stage3_uncond/ -r last      # after third stage 
```
A `recon` folder in the config directory will contain the `.ply` reconstructions. `max_batch` arguments in `test.py` are used for running marching cubes; change it to the max value your GPU memory can hold.



## References
We adapt code from <br>
GenSDF https://github.com/princeton-computational-imaging/gensdf <br>
DALLE2-pytorch https://github.com/lucidrains/DALLE2-pytorch <br>
Convolutional Occupancy Networks https://github.com/autonomousvision/convolutional_occupancy_networks (for PointNet encoder) <br>
Multimodal Shape Completion via cGANs https://github.com/ChrisWu1997/Multimodal-Shape-Completion (for conditional metrics) <br>
PointFlow https://github.com/stevenygd/PointFlow (for unconditional metrics)
