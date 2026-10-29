# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Video](https://www.youtube.com/watch?v=oaHCtB6yiKU) | [Surfel Rasterizer (CUDA)](https://github.com/hbb1/diff-surfel-rasterization) | [Surfel Rasterizer (Python)](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing) | [DTU+COLMAP (3.5GB)](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) | [SIBR Viewer Pre-built for Windows](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor/releases/download/v1.0/GS_Monitor.zip) | [Web Viewer](https://github.com/mkkellogg/GaussianSplats3D) <br>

![Teaser image](assets/teaser.jpg)

This repo contains the official implementation for the paper "2D Gaussian Splatting for Geometrically Accurate Radiance Fields". Our work represents a scene with a set of 2D oriented disks (surface elements) and rasterizes the surfels with [perspective correct differentiable raseterization](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing). Our work also develops regularizations that enhance the reconstruction quality. We also devise meshing approaches for Gaussian splatting.


## ⭐ New Features 
- 2025/12/19: Our work is featured in an in-depth blog post on [LearnOpenCV](https://learnopencv.com/)! Thanks to [Shubham Anand](https://www.linkedin.com/in/shubham-anand-91a10b211/).
- 2024/07/20: Web-based viewer [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) also supports 2DGS. Thanks to [Mark Kellogg](https://github.com/mkkellogg).
- 2024/07/19: [Colab Notebook](https://github.com/atakan-topaloglu/2d_gaussian_splatting_colab) is supported! Thanks to [atakan-topaloglu](https://github.com/atakan-topaloglu)
- 2024/06/10: [SIBR Viewer](https://github.com/RongLiu-Leo/2d-gaussian-splatting) is supported! Thanks to [Rong](https://github.com/RongLiu-Leo/).
- 2024/06/05: [Remote Viewer](https://github.com/hwanhuh/2D-GS-Viser-Viewer) based on Viser is supported! Thanks to [HwanHeo](https://github.com/hwanhuh).
- 2024/05/30:  Fixed a bug related to unbounded meshing. The foreground mesh quality should now be consistent with the bounded mesh.
- 2024/05/17: Improve training speed by 30%~40% through the [cuda operator fusing](https://github.com/hbb1/diff-surfel-rasterization/pull/7). Please update the diff-surfel-rasterization submodule if you have already installed it. 
    ```bash
    git submodule update --remote  
    pip install submodules/diff-surfel-rasterization
    ```
- 2024/05/05: Important updates - Now our algorithm supports **unbounded mesh extraction**!
Our key idea is to contract the space into a sphere and then perform **adaptive TSDF truncation**. 

![visualization](assets/unbounded.gif)

## 🎓 Community Resources & Tutorials

- **gsplat library documentation** (official rasterization API reference)  
  https://docs.gsplat.studio/main/apis/rasterization.html#id1

- **GaussianSplats3D** – Popular WebGL/Three.js viewer with strong community support  
  https://github.com/mkkellogg/GaussianSplats3D

- **SuperSplat** – High-performance WebGPU viewer by PlayCanvas  
  https://github.com/playcanvas/supersplat

- **In-Depth Practitioner Guide** – Comprehensive tutorial on the full 2D Gaussian Splatting pipeline (theory to implementation) by Shubham Anand  
  [LearnOpenCV Blog Post](https://learnopencv.com/2d-gaussian-splatting/)

- **Diff-Surfel-Tracing** - A differentiable ray-tracing implementation built on the surfel representation by [xbillowy](https://github.com/xbillowy)
   https://github.com/xbillowy/diff-surfel-tracing

## SIBR Viewer


https://github.com/RongLiu-Leo/2d-gaussian-splatting/assets/102014841/b75dd9a7-e3ee-4666-99ff-8c9121ff66dc


The Pre-built Viewer for Windows can be found [here](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor/releases/download/v1.0/GS_Monitor.zip). If you use Ubuntu or want to check the viewer usage, please refer to [GS Monitor](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor).
### How to use
Firstly open the viewer, 
```shell
<path to downloaded/compiled viewer>/bin/SIBR_remoteGaussian_app_rwdi
```
and then
```shell
# Monitor the training process
python train.py -s <path to COLMAP or NeRF Synthetic dataset> 
# View the trained model
python view.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model> 
```

## Installation

```bash
# download
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

# if you have an environment used for 3dgs, use it
# if not, create a new environment
conda env create --file environment.yml
conda activate surfel_splatting
```
## Training
To train a scene, simply use
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
Training logs scalar losses only; it does not run periodic validation or generate
TensorBoard image previews. PLY saving and checkpoint saving are unchanged.
`--test_iterations` is accepted for compatibility with existing commands but is
ignored. Use `render.py` and `metrics.py` separately when evaluation is needed.

To train with images loaded in memory-budgeted batches:
```bash
python train.py -s <path to dataset> --data_cache_mb 1024
```
`--data_cache_mb` accepts a non-negative number in MiB (1 MiB = 1024² bytes).
The default, `0`, keeps the original full-data loading behavior. With a positive
budget, COLMAP and Blender images are decoded on demand. Each batch is sized by
the resized floating-point image tensors, including alpha masks and enabled
inverse-depth maps/validity masks, rather than
compressed file sizes. Training still uses one random view per optimizer step;
every view is visited once per shuffled epoch before starting the next epoch.
Completed batches are released before loading the next batch. If the entire
training set fits, it stays cached across epochs. An image larger than the budget
is loaded alone with a warning; use a larger budget or a lower image resolution
to avoid this exception.

The budget covers cached training pixels, not total process RAM or GPU memory:
model/optimizer state, camera matrices, temporary image decoding buffers and one
validation view require additional memory. Validation images are loaded one at a
time. RGB tensors remain on CPU until used by training, and alpha masks retain
the existing `--data_device` behavior. The setting is saved in `cfg_args`; render
scripts also load cameras on demand when enabled (use `--data_cache_mb 0` to
restore eager loading).

### Depth supervision

Enable depth supervision alongside batching with:
```bash
python train.py -s <path to dataset> --depths depths --data_cache_mb 1024 \
    --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
```
`--depths` (or `-d`) is a directory, relative to the dataset root or absolute.
Omitting it keeps RGB-only training. COLMAP datasets use the
[3DGS depth calibration format](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py):
```text
dataset/
  images/frame.001.jpg
  depths/frame.001.png
  sparse/0/depth_params.json
```
Example `depth_params.json`:
```json
{
  "frame.001": {"scale": 0.8, "offset": 0.02},
  "frame.002": {"scale": 0.9, "offset": -0.01}
}
```
Keys are COLMAP image names with only the final extension removed, preserving
relative subdirectories (e.g. `camera1/frame.001`). Depth files use the same key
with a `.png` extension. Supply **single-channel uint16 inverse-depth PNGs**,
not metric-depth images or color visualizations. The loader divides PNG values
by **65536**, resizes to the training resolution with bilinear interpolation,
then applies `inverse_depth = normalized_png * scale + offset`. JSON is read
once; the median of positive scales is computed automatically. Views with
non-positive scales or scales outside `[0.2 * median, 5 * median]` are skipped.
Missing JSON entries warn and skip that view; missing/malformed JSON or missing/
invalid depth files for eligible views raise an error. An entirely ineligible
training set raises an error when depth supervision is enabled.

For Blender/NeRF Synthetic datasets, the existing `transforms_train.json` and
`transforms_test.json` camera loading is retained. Following 3DGS, depth files
are `<depths>/<image_basename>.png`, decoded as inverse depth using a **512**
divisor, without a COLMAP calibration JSON.

The loss compares the reciprocal of 2DGS `surf_depth` (selected by
`--depth_ratio`) with the calibrated inverse depth. It uses masked L1 averaged
over all image pixels. Non-positive/non-finite target depths and transparent
pixels in images with an alpha mask are excluded. The two depth weights default
to `1.0` and `0.01`, with exponential decay over `--iterations`; set both to `0`
to disable the loss. Progress output includes `depth`, and TensorBoard records
`train_loss_patches/depth_loss` and `train_loss_patches/depth_weight`.

Depth and mask tensors remain on CPU until their view is used, occupy 8 bytes
per resized pixel together, and count toward `--data_cache_mb`. They are loaded
and released with their RGB batch. This also works with eager loading (`0`),
validation and checkpoint resume. The depth directory is saved in `cfg_args`;
pass `--depths ""` to render an existing model without loading supervision files.

Commandline arguments for regularizations
```bash
--lambda_normal  # hyperparameter for normal consistency
--lambda_distortion # hyperparameter for depth distortion
--depth_ratio # 0 for mean depth and 1 for median depth, 0 works for most cases
```
**Tips for adjusting the parameters on your own dataset:**
- For unbounded/large scenes, we suggest using mean depth, i.e., ``depth_ratio=0``,  for less "disk-aliasing" artifacts.

## Testing
### Bounded Mesh Extraction
To export a mesh within a bounded volume, simply use
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> 
```
To fuse views in memory-budgeted batches, use the same `--data_cache_mb` option
as training (in MiB):
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> \
    --data_cache_mb 1024 --skip_train --skip_test
```
With a positive budget, bounded extraction loads camera images in their original
order, renders and integrates each view into one shared TSDF volume, then releases
the view and its rendered RGB/depth maps. It does not cache all rendered maps.
Global bounds and default mesh parameters still use all camera poses. Batch sizes
use the same resized image/alpha/depth tensor accounting as training; a view larger
than the budget is loaded alone with a warning. The budget excludes the Gaussian
model, TSDF volume, mesh, render buffers and decoding temporaries, so it is not a
total RAM/VRAM limit. `--data_cache_mb 0` keeps the cached extraction path.
The skip flags above avoid the separate train/test image exports, which still
cache rendered maps; `--unbounded` also retains its existing render cache.

Each loaded batch prints `[Camera batch N]` with its 1-based view range, image
count and estimated pixel-data size versus the budget. After fusion, the total
number of views and batches is printed. The TSDF progress bar still tracks all
views across batches. A final batch count of one means only one batch was needed;
lower `--data_cache_mb` to use smaller batches. Reported sizes are the tensor
budget estimates, not measured process RAM/VRAM usage.

Use `--mesh_output_dir /path/to/meshes` to save both `fuse.ply` and
`fuse_post.ply` in a custom directory, which is created automatically. This also
applies to `--unbounded` (`fuse_unbounded.ply` and `fuse_unbounded_post.ply`).
Relative paths are resolved from the current working directory. If omitted,
meshes are saved under `<model_path>/train/ours_<iteration>` as before.

Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion, use
```bash
--depth_ratio # 0 for mean depth and 1 for median depth
--voxel_size # voxel size
--depth_trunc # depth truncation
```
If these arguments are not specified, the script will automatically estimate them using the camera information.
### Unbounded Mesh Extraction
To export a mesh with an arbitrary size, we devised an unbounded TSDF fusion with space contraction and adaptive truncation.
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> --mesh_res 1024
```

## Quick Examples
Assuming you have downloaded [MipNeRF360](https://jonbarron.info/mipnerf360/), simply use
```bash
python train.py -s <path to m360>/<garden> -m output/m360/garden
# use our unbounded mesh extraction!!
python render.py -s <path to m360>/<garden> -m output/m360/garden --unbounded --skip_test --skip_train --mesh_res 1024
# or use the bounded mesh extraction if you focus on foreground
python render.py -s <path to m360>/<garden> -m output/m360/garden --skip_test --skip_train --mesh_res 1024
```
If you have downloaded the [DTU dataset](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), you can use
```bash
python train.py -s <path to dtu>/<scan105> -m output/date/scan105 -r 2 --depth_ratio 1
python render.py -r 2 --depth_ratio 1 --skip_test --skip_train
```
**Custom Dataset**: We use the same COLMAP loader as 3DGS, you can prepare your data following [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes). 

> [!WARNING] 
> In our **preprocessed DTU dataset**, we store the mask in the alpha channel. When using the **DTU dataset** in the [gaussian-splatting repository](https://github.com/graphdeco-inria/gaussian-splatting), please note that the background may be masked. To train DTU with background, we have commented [these lines](https://github.com/hbb1/2d-gaussian-splatting/blob/df1f6c684cc4e41a34937fd45a7847260e9c6cd7/scene/cameras.py#L43C1-L48C38) out.

## Full evaluation
We provide scripts to evaluate our method of novel view synthesis and geometric reconstruction.
<details>
<summary><span style="font-weight: bold;">Explanation of Performance Differences to the Paper</span></summary>

We have re-implemented the repository for improved efficiency, which has slightly impacted performance compared to the original paper. Two factors have influenced this change:

- 📈 We fixed some minor bugs, such as a half-pixel shift in TSDF fusion, resulting in improved geometry reconstruction.

- 📉 We removed the gradient of the low-pass filter used for densification, which reduces the number of Gaussians. As a result, the PSNR has slightly dropped, but we believe this trade-off is worthwhile for real-world applications.

You can report either the numbers from the paper or from this implementation, as long as they are discussed in a comparable setting.
</details>

#### Novel View Synthesis
For novel view synthesis on [MipNeRF360](https://jonbarron.info/mipnerf360/) (which also works for other colmap datasets), use
```bash
python scripts/m360_eval.py -m60 <path to the MipNeRF360 dataset>
```

#### Geometry reconstruction
For geometry reconstruction on DTU dataset, please download the preprocessed data from [Drive](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) or [Hugging Face](https://huggingface.co/datasets/dylanebert/2DGS). You also need to download the ground truth [DTU point cloud](https://roboimagedata.compute.dtu.dk/?page_id=36). 
```bash
python scripts/dtu_eval.py --dtu <path to the preprocessed DTU dataset>   \
     --DTU_Official <path to the official DTU dataset>
```
We provide <a> Evaluation Results (Pretrained, Meshes)</a>. 
<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

Chamfer distance on DTU dataset (lower is better)

|   | 24   | 37   | 40   | 55   | 63   | 65   | 69   | 83   | 97   | 105  | 106  | 110  | 114  | 118  | 122  | Mean |
|----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| Paper    | 0.48 | 0.91 | 0.39 | 0.39 | 1.01 | 0.83 | 0.81 | 1.36 | 1.27 | 0.76 | 0.70 | 1.40 | 0.40 | 0.76 | 0.52 | 0.80 |
| Reproduce | 0.46 | 0.80 | 0.33 | 0.37 | 0.95 | 0.86 | 0.80 | 1.25 | 1.24 | 0.67 | 0.67 | 1.24 | 0.39 | 0.64 | 0.47 | 0.74 |
</details>

For geometry reconstruction on TnT dataset, please download the preprocessed [TnT_data](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main). You also need to download the ground truth [TnT_GT](https://www.tanksandtemples.org/download/), including ground truth point cloud, alignments and cropfiles.

> [!IMPORTANT]  
> Due to historical issue, you should use open3d==0.10.0 for evaluating TNT.

```bash
# use open3d 0.18.0, skip metrics
python scripts/tnt_eval.py --TNT_data <path to the preprocessed TNT dataset>   \
     --TNT_GT <path to the official TNT evaluation dataset> --skip_metrics

# use open3d 0.10.0, skip traing and rendering
python scripts/tnt_eval.py --TNT_data <path to the preprocessed TNT dataset>   \
     --TNT_GT <path to the official TNT evaluation dataset> --skip_training --skip_rendering
```
<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

F1 scores on TnT dataset (higher is better)

|    | Barn   | Caterpillar | Ignatius | Truck  | Meetingroom | Courthouse | Mean | 
|--------|--------|-------------|----------|--------|-------------|------------|------------|
| Reproduce | 0.41  | 0.23      | 0.51   | 0.45 | 0.17      | 0.15      | 0.32 |
</details>


## FAQ
- **Training does not converge.**  If your camera's principal point does not lie at the image center, you may experience convergence issues. Our code only supports the ideal pinhole camera format, so you may need to make some modifications. Please follow the instructions provided [here](https://github.com/graphdeco-inria/gaussian-splatting/issues/144#issuecomment-1938504456) to make the necessary changes. We have also modified the rasterizer in the latest [commit](https://github.com/hbb1/diff-surfel-rasterization/pull/6) to support data accepted by 3DGS. To avoid further issues, please update to the latest commit.

- **No mesh / Broken mesh.** When using the *Bounded mesh extraction* mode, it is necessary to adjust the `depth_trunc` parameter to perform TSDF fusion to extract meshes. On the other hand, *Unbounded mesh extraction* does not require tuning the parameters but is less efficient.  

- **Can 3DGS's viewer be used to visualize 2DGS?** Technically, you can export 2DGS to 3DGS's ply file by appending an additional zero scale. However, due to the inaccurate affine projection of 3DGS's viewer, you may see some distorted artefacts. We are currently working on a viewer for 2DGS, so stay tuned for updates.

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The TSDF fusion for extracting mesh is based on [Open3D](https://github.com/isl-org/Open3D). The rendering script for MipNeRF360 is adopted from [Multinerf](https://github.com/google-research/multinerf/), while the evaluation scripts for DTU and Tanks and Temples dataset are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively. The fusing operation for accelerating the renderer is inspired by [Han's repodcue](https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce). We thank all the authors for their great repos. 


## Citation
If you find our code or paper helps, please consider citing:
```bibtex
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```

## Black-mask voxel densification

Enable fixed voxel guidance from the **black** regions of `mask_tower`:

```bash
python train.py -s /path/to/colmap -m output/black_voxel \
  --voxel_densify --voxel_masks mask_tower \
  --voxel_black_threshold 0 --voxel_size 0.05 --voxel_dilation 2 \
  --voxel_inside_grad_threshold 0.0002 \
  --voxel_outside_grad_threshold 0.0001
```

`--voxel_densify` defaults to off. Both spatial gradient thresholds default to
`0.0002`; either may be larger. Lower thresholds make more points eligible for
clone/split. With guidance enabled, these two values replace
`--densify_grad_threshold`. The example applies a higher threshold inside the
black-mask grid and a lower threshold outside it. Names describe grid membership,
not foreground/background semantics.

The mask path is relative to `-s`, or absolute. For registered image
`subdir/frame.jpg`, supply `mask_tower/subdir/frame.png`. Masks must match the
original image and COLMAP camera dimensions, independently of training `-r`.
A point becomes a seed only if it has at least one checked COLMAP feature
observation and **every** checked observation falls on a pixel whose grayscale
value is `<= --voxel_black_threshold` (default 0, pure black; range 0–254).
This is not simply the complement of the all-white selection: mixed observations,
invalid/out-of-image coordinates and unobserved points do not become seeds.
Only registered images present in `--images` are checked, including evaluation
views. Missing masks, size mismatches and empty seed selections raise errors.
Binary and text COLMAP models are supported. Initial PLY coordinates must match
the original COLMAP points at float32 precision; reordering and subsets are
supported, with conservative handling of duplicate coordinates.

The training point cloud remains complete, with its original colors. Seeds only
initialize an immutable sparse grid in COLMAP world units. Dilation uses a cube
of neighbouring cells, queried implicitly without allocating a dense grid.
During densification, current Gaussian centers determine which threshold applies.
A white-mask point can still lie inside a dilated black-mask grid, so the startup
and densification logs report actual spatial membership.

Clone/split scale conditions, parent removal, opacity reset, pruning (including
`max_radii2D` clearing), losses and full-image supervision retain the `main`
behavior. This implementation adds no population caps or opacity grace schedule.
With equal inside/outside thresholds, its densification and pruning match the
original scalar-threshold path on the same inputs.

New checkpoints save the fixed grid with a black-mask format tag. Resuming one
restores that grid without rereading masks; saved voxel size/dilation take
precedence, while current CLI gradient thresholds apply. Legacy 12-field
checkpoints build a grid from the original input point cloud before restoring
trained centers. The old white-mask branch's grid format is rejected explicitly.
PLY outputs and input COLMAP files are unchanged.
