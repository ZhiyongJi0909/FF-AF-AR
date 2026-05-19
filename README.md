# FF-UA-MF: Feature Fluctuation-Driven Uncertainty-Aware Method for Multimodal Fusion

Official implementation of **"Feature Fluctuation-Driven Uncertainty-Aware Method for Multimodal Fusion in Special-Operations Monitoring"**.

## Overview

This repository contains the PyTorch implementation of FF-UA-MF, a lightweight uncertainty-aware multimodal fusion framework designed for industrial special-operations monitoring on edge devices. The model dynamically re-weights visual and inertial modalities based on real-time feature fluctuation analysis, without requiring explicit noise priors or Bayesian posterior sampling.

## Key Features

- **Lightweight dual-stream architecture** (3.35M parameters, 3.1 GFLOPs)
- **Feature Fluctuation-Driven Uncertainty-Aware (FFUA)** fusion: uses temporal variance and decision dispersion as reliability proxies
- **Joint multitask loss**: classification + InfoNCE alignment with same-class masking + Frobenius-norm decorrelation
- **Edge-friendly**: 76 FPS inference throughput on standard hardware
- **High accuracy**: 98.66% average accuracy on UTD-MHAD and SO-MHAD benchmarks

## Architecture

```
Input (Video + IMU)
    │
    ├── VSOARnet ──→ [frame_features, pooled_features] ──┐
    │       (RepViT-based visual extractor)               │
    │                                                     │
    └── ISOARnet ──→ [frame_features, pooled_features] ──┤
            (CNN-BiGRU inertial extractor)                │
                                                          ▼
                                                    FFUA Fusion Module
                                                     (Uncertainty MLPs
                                                     + Inverse-variance weighting)
                                                          │
                                                          ▼
                                                 Final Classification
```

### Model Components

| Paper Name | Code File | Description |
|:---|:---|:---|
| **VSOARnet** | `TMAInet_model_v1.py` | Video-based Special Operations Action Recognition network. Lightweight RepViT with 3D stem, neighborhood-aware gated channel enhancement (CFF + ECA), and reparameterized training |
| **ISOARnet** | `imu_model_v4.py` | Inertial-based Special Operations Action Recognition network. Multiscale 1D CNN + Bi-GRU + channel attention |
| **FFUA** | `fusion_model.py` | Feature Fluctuation-Driven Uncertainty-Aware fusion module |
| **FF_UA_MF** | `fusion_model.py` | Complete multimodal fusion model |

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- torchvision
- timm
- opencv-python
- scipy
- pywt (PyWavelets)
- numpy
- pandas
- tqdm
- matplotlib
- wandb (optional, for experiment tracking)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FF-UA-MF.git
cd FF-UA-MF

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### UTD-MHAD Dataset

Download the [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) dataset and organize as follows:

```
UTD-MHAD/
├── class_name_1/
│   ├── depth/
│   ├── inertial/
│   └── RGB/
│       ├── aX_sY_tZ_color/
│       │   ├── image_00001.jpg
│       │   └── n_frames
├── class_name_2/
│   └── ...
└── utdTrainTestlist/
    └── classInd.txt
```

### SO-MHAD Dataset (Self-collected)

Follow the same directory structure as UTD-MHAD for the self-collected special-operations dataset.

## Training

### Stage 1: Pre-train VSOARnet on UCF101

```bash
# Pre-train the visual backbone
python pretrain_vsoarnet.py \
    --dataset ucf101 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

### Stage 2: End-to-End Fusion Training

```bash
python code/main.py \
    --dataRoot /path/to/UTD-MHAD \
    --pretrainedVsoarnet /path/to/best_vsoarnet_ucf101_3d.pth \
    --batchSize 8 \
    --epochs 100 \
    --lr 5e-5 \
    --imu-lr 2.5e-4 \
    --weight-decay 5e-4 \
    --log-dir ./logs_fusion
```

### Training Arguments

| Argument | Default | Description |
|:---|:---|:---|
| `--dataRoot` | (required) | Dataset root directory |
| `--pretrainedVsoarnet` | `None` | Path to VSOARnet pre-trained weights |
| `--batchSize` | `8` | Training batch size |
| `--epochs` | `100` | Total training epochs |
| `--lr` | `5e-5` | VSOARnet learning rate |
| `--imu-lr` | `2.5e-4` | ISOARnet learning rate |
| `--weight-decay` | `5e-4` | Weight decay |
| `--seed` | `42` | Random seed |
| `--use-wandb` | `False` | Enable Weights & Biases logging |

## Model Weights

Pre-trained model weights will be released upon paper acceptance. The model supports loading partial state dicts for fine-tuning.

## Quick Test

```bash
cd code
python test_ffua.py
```

This runs a forward-pass sanity check verifying:
- Correct tensor shapes for all outputs
- Valid uncertainty weights (sum to 1)
- Proper loss computation

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{ffuamf2024,
  title={Feature Fluctuation-Driven Uncertainty-Aware Method for Multimodal Fusion in Special-Operations Monitoring},
  author={Your Name and Co-authors},
  journal={},
  year={2024}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- UTD-MHAD dataset: [The University of Texas at Dallas](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
- RepViT architecture inspired by [RepViT: Revisiting Mobile CNN From ViT Perspective](https://arxiv.org/abs/2307.09283)

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
