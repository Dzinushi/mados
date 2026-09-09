![MADOS Logo](./.github/MADOS_LOGO_text.png)


[[`dataset`](https://zenodo.org/records/10664073)]

Marine Debris and Oil Spill (MADOS) is a marine pollution dataset based on Sentinel-2 remote sensing data, focusing on marine litter and oil spills. Other sea surface features that coexist with or have been suggested to be spectrally similar to them have also been considered. MADOS formulates a challenging semantic segmentation task using sparse annotations.

 In order to download MADOS go to https://doi.org/10.5281/zenodo.10664073.

 ## Extensions: TReLU + Adaptive Norms (ALN / ABN / AGN) + Auto-Pruning

 This fork extends the original MariNeXt (`marinext/configs/marinext.tiny.240x240.mados_original.py`: GELU + SyncBN + GN + LayerNorm) with the following changes:

 1. **TReLU activation instead of GELU.** Learnable piecewise-linear `y = x * (r1 * [x>=0] + r2 * [x<0])`, `r1/r2` are `nn.Parameter` (init `r1=1.0, r2=0.01`). Applied in backbone (`MSCAN`: `Mlp`, `StemConv`, `SpatialAttention` via `act_layer`) and decoder (`LightHamHead` via `act_cfg`). See `marinext/mmseg/models/backbones/mscan.py`, `marinext/mmseg/models/custom_conv_module.py`, `marinext/mmseg/models/decode_heads/ham_head.py`.
 2. **ALN — AdaptiveLayerNorm instead of LayerNorm.** Replaces final `nn.LayerNorm(embed_dims[i])` of each MSCAN stage: normalize, then asymmetric gating by sign (`w1/w2 + bias`, init `l1=l2=1.0`). Enabled via `layer_norm=dict(type="AdaptiveLayerNorm", l1=1.0, l2=1.0)`.
 3. **ABN — AdaptiveBatchNorm instead of SyncBN.** Replacement for backbone `norm_cfg`, with `running_mean/var` and same sign-gated `w1/w2 + bias`. Enabled via `norm_cfg=dict(type="ABN", l1=1.0, l2=1.0)`.
 4. **AGN — AdaptiveGroupNorm instead of GN.** Replacement for decoder `GN(num_groups=32)` in `ConvModule`: group-wise normalization + sign-gated `w1/w2 + bias`. Enabled via `norm_cfg=dict(type="AGN", num_groups=32, l1=1.0, l2=1.0)`.
 5. **Config matrix (`marinext/configs/`):** `mados_original.py` = baseline; `mados.py` = TReLU-only; `mados_aln.py` / `mados_aln_abn.py` / `mados_aln_agn.py` / `mados_aln_abn_agn.py` = combinations.
 6. **`--config` plumbing:** `MariNext(..., cfg_name)`, `train.py` and `evaluation.py` accept `--config` so any config above can be selected from CLI.
 7. **Automatic TReLU pruning + cost report:** `marinext/trelu_prune_eval.py:apply_dynamic_pruning()` replaces `Block.mlp` with `ZeroLayer` when `max(|r1|,|r2|) < prune_threshold (1e-20)` and optionally saves `*_pruned.pth`. `marinext/model_summary.py` (via `thop`) reports params / FLOPs / latency before/after pruning.
 8. **Repro scripts:** `train.sh`, `eval.sh`, `trelu_prune_eval.sh`, `model_summary.sh`.

 ### Our Paper Results (Metrics)

 | Metric   | Vanilla | Vanilla (our) | TReLU | TReLU + ALN | TReLU + ALN + AGN | TReLU + ALN + AGN (no L2) |
 |----------|---------|---------------|-------|-------------|-------------------|---------------------------|
 | mRec     | —       | 76.7          | 81.1  | 84.2        | 85.0              | 87.0                      |
 | mIoU     | 64.3    | 64.5          | 70.3  | 72.9        | 74.3              | 76.8                      |
 | OA       | 89.1    | 92.2          | 94.0  | 94.2        | 93.6              | 93.6                      |
 | F1-macro | 76.0    | 75.6          | 81.4  | 83.2        | 84.3              | 86.1                      |


### Our Paper Results (Pruning)
|                      | TReLU + ALN + AGN | TReLU + ALN + AGN (no L2) |
|----------------------|-------------------|---------------------------|
| File Size (Mb)       | 16.4              | 7.40                      |
| Trainable Params (M) | 4.24              | 1.89                      |
| GFLOPs               | 4.49              | 3.50                      |
| FPS                  | 76.7              | 94.4                      |
| Peak GPU Memory (Gb) | 0.07              | 0.06                      |


 ### Reproduce: train / eval / prune

 Default recommended run is **with weight decay (`--decay 1e-4`) + TReLU**, because pruning only works in this regime:

 ```bash
 # 1. Train (pruning-enabled, default):
 python -m marinext.train --epochs 120 --reduce_lr_on_plateau 1 --factor 0.7 --patience 5 --batch 8 --num_workers 6 --decay 1e-4 --path ./data/MADOS/ --config marinext.tiny.240x240.mados_aln_abn_agn.py
 # see train.sh

 # 2. Standard eval:
 # eval.sh:
 # config="marinext.tiny.240x240.mados_aln_abn.py"
 # model_path=".../original"
 # python -m marinext.evaluation --path ./data/MADOS/ --model_path $model_path --config $config --split "test"
 bash eval.sh

 # 3. Eval with automatic pruning (TReLU models only):
 # trelu_prune_eval.sh:
 # python -m marinext.trelu_prune_eval --path ./data/MADOS/ --model_path $model_path --config $config --prune_threshold 1e-20 --split "test"
 bash trelu_prune_eval.sh

 # 4. Cost of pruning:
 bash model_summary.sh
 ```

 Notes on `decay`:

 - `--decay 0.0` (as currently in `train.sh`) gave the best raw accuracy, but pruning is impossible there — `r1/r2` never collapse to zero without L2.
 - `--decay 1e-4` (default in `train.py`) enables `r1/r2 → 0` and therefore `MLP → ZeroLayer` pruning at `1e-20`.
 - Suggested workflow to get both: (a) train with `--decay 1e-4`, prune, record dropped `Block.mlp` layers; (b) retrain from scratch with `--decay 0.0` and those layers removed — this may retain accuracy while keeping the compression. Not yet automated in this fork.
 - Optimizer note: training uses `Adam(weight_decay=...)`. `AdamW` would be cleaner for L2-decoupling of `r1/r2`, but was intentionally not introduced to keep the diff minimal against the original fork and only demonstrate feasibility of the idea.

 ## Installation
 
 ```bash
conda create -n mados python=3.8.12

conda activate mados

conda install -c conda-forge gdal==3.3.2

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

conda install pytables==3.7.0
```

 ### If drop error with 'libtorch_cpu.so' or 'libtorch_cuda.so' use next steps
```bash
execstack -c $CONDA_PREFIX/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so
```
 ### Next check if all right. If not may be pip packages are bad interacting with conda packages. Try to use only conda-packages. At first remove pip-packages (fail to applying execstack):
```bash
python -m pip uninstall -y torch torchvision torchaudio
```
 ### Install from conda (CUDA-version if needed)
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio cudatoolkit=11.3 -c pytorch -y
```
 ### Find libtorch_cpu.so in conda-mados (path in command below for example)
```bash
execstack -c $CONDA_PREFIX/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so
```
 ### Re-install the library if the related error occurs
```bash
pip install -U --force-reinstall charset-normalizer
```

 ## Evaluate MariNeXt

To evaluate MariNeXt, place the MADOS dataset under the `data` folder, download the pretrained models (5 different runs) from [here](https://drive.google.com/drive/folders/1VwkFp47TEvRVXHNbucBmmylfZwIUmCWx?usp=drive_link) and place them under the `marinext/trained_models` folder and then run the following:

```bash
python marinext/evaluation.py --path ./data/MADOS --model_path marinext/trained_models/1
```

 ## Train MariNeXt

 To train MariNeXt from scratch, run the following:


 ```bash
 python marinext/train.py --path ./data/MADOS
 ```

 ## Stack Patches

To stack the image patches to form multispectral images, run the following:


```bash
python utils/stack_patches.py --path ./data/MADOS
```

 ## Spectral Signatures Extraction
To extract the spectal signatures of MADOS dataset (after stacking) and store them in a HDF5 Table file (DataFrame-like) run the following:

```bash
python utils/spectral_extraction.py --path ./data/MADOS_nearest
```

Alternatively, you can download the `dataset.h5` file from [here](https://drive.google.com/file/d/1BUIxcm1SLU9sqr8NE2FKJvJJPv2RLyk-/view?usp=sharing).

To load the `dataset.h5`, run in a python cell the following:

```python
import pandas as pd

hdf = pd.HDFStore('./data/dataset.h5', mode = 'r')

df_train = hdf.select('Train')
df_val = hdf.select('Validation')
df_test = hdf.select('Test')

hdf.close()
```


 ## Acknowledgment

This implementation is mainly based on [MARIDA](https://github.com/marine-debris/marine-debris.github.io), [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt), [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [Segformer](https://github.com/NVlabs/SegFormer) and [Enjoy-Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger).



If you find this repository useful, please consider giving a star :star: and citation:
 > Kikaki K., Kakogeorgiou I., Hoteit I., Karantzalos K. Detecting Marine Pollutants and Sea Surface Features with Deep Learning in Sentinel-2 Imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 2024.
 
