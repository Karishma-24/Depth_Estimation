# Monocular Depth Estimation for AR Scene Understanding

A deep learning project for monocular depth estimation from single RGB images, targeting indoor Augmented Reality scene understanding. Built and evaluated as part of a Deep Learning course at BITS Pilani, Dubai Campus.

---

## Overview

This project implements and evaluates a monocular depth estimation pipeline for indoor AR applications. Given a single RGB image, the model produces a dense depth map that can be used to understand the 3D spatial structure of a scene — enabling accurate object placement, occlusion handling, and scene interaction in AR systems.

The motivation: traditional depth sensing hardware (LiDAR, stereo cameras) is expensive and unsuitable for lightweight AR devices. Monocular depth estimation from a single RGB image offers a practical, low-cost alternative.

---

## Model Architecture

The final model (**V8**) is a **ResNet50 U-Net** encoder-decoder network trained on the NYU Depth V2 indoor dataset.

### Encoder
- **Backbone**: ResNet50 (ImageNet pretrained)
- Leverages deep residual features across 5 stages
- Bottleneck blocks produce skip connections at each stage for the decoder

### Decoder
- Custom U-Net decoder with transposed convolutions for upsampling
- Skip connections from encoder stages fuse low-level spatial detail with high-level semantic features
- Output: single-channel depth map at the input resolution

### Training Configuration
| Parameter | Value |
|---|---|
| Input resolution | 256 × 256 |
| Optimizer | Adam (differential LR) |
| Encoder LR | 1e-4 |
| Decoder LR | 1e-3 |
| Loss function | SSIM + L1 + Gradient |
| Early stopping | ✓ |
| Training dataset | NYU Depth V2 (indoor) |

### Loss Function
A multi-component loss combining:
- **SSIM loss** — structural similarity for perceptual quality
- **L1 loss** — pixel-wise absolute error for accuracy
- **Gradient loss** — edge sharpness and depth boundary preservation

---

## Results

Evaluated on the NYU Depth V2 test set:

| Metric | Score |
|---|---|
| δ1 (% < 1.25) | 0.6381 |
| δ2 (% < 1.25²) | 0.8967 |
| δ3 (% < 1.25³) | 0.9650 |
| AbsRel | 0.2405 |
| SqRel | 0.0423 |
| RMSE | 0.1335 |
| RMSElog | 0.2883 |

---

## Dataset

**NYU Depth V2** — indoor RGB-D dataset captured with a Microsoft Kinect.

- 1,449 densely labelled RGB-depth pairs
- 464 indoor scenes across kitchens, bedrooms, classrooms, and more
- 407,024 unlabeled frames
- Used via the [Kaggle version by soumikrakshit](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)

Training PNGs are 8-bit; test PNGs are 16-bit (handled accordingly in the data pipeline).

---

## Web Application

A Flask-based web app supports real-time video inference with side-by-side RGB and depth map visualization.

### Stack
- **Backend**: Flask + Flask-CORS
- **Frontend**: React 19 + TypeScript (served statically)
- **Video writing**: `imageio` with `ffmpeg` subprocess fallback

### Running the App

```bash
# Activate environment
conda activate depth_venv

# Start Flask server
python app.py
```

The app will be available at `http://localhost:8080`. Upload a video or image to get a depth map output.

---

## Project Structure

```
.
├── app.py                  # Flask backend for inference
├── model/
│   └── v8_checkpoint.pth   # Trained model weights
├── src/
│   ├── model.py            # ResNet50 U-Net architecture
│   ├── dataset.py          # NYU Depth V2 data loader
│   ├── loss.py             # SSIM + L1 + Gradient loss
│   └── metrics.py          # Evaluation metrics
├── frontend/               # React + TypeScript frontend
├── notebooks/              # Kaggle training notebooks
└── README.md
```

---

## Setup

### Requirements

- Python 3.8
- PyTorch
- torchvision
- OpenCV
- Flask, Flask-CORS
- imageio
- matplotlib
- ffmpeg (via conda-forge)

### Installation

```bash
# Create and activate conda environment
conda create -n depth_venv python=3.8
conda activate depth_venv

# Install dependencies
pip install torch torchvision flask flask-cors imageio matplotlib opencv-python

# Install ffmpeg
conda install -c conda-forge ffmpeg
```

---

## Training

Training was conducted on Kaggle (T4 x2 GPUs). To reproduce:

1. Upload the NYU Depth V2 dataset to Kaggle
2. Run the training notebook via **Save & Commit** for headless GPU execution
3. Checkpoints are saved incrementally; resume by uploading a checkpoint as a Kaggle dataset

Key practices used during training:
- Explicit `del` of tensors + `torch.cuda.empty_cache()` + `gc.collect()` after each epoch to prevent OOM
- Differential learning rates: lower LR for pretrained encoder, higher for randomly initialized decoder
- Early stopping to prevent overfitting

---

## References

1. Kim et al. (2021). *A Hybrid Approach to Industrial AR Using Deep Learning-Based Facility Segmentation and Depth Prediction.* Sensors, 21(1), 307.
2. Zhou et al. (2022). *Learning Depth Estimation From Memory Infusing Monocular Cues.* IEEE Access, 10, 21359–21369.
3. Lahiri et al. (2024). *Deep Learning-Based Stereopsis and Monocular Depth Estimation Techniques: A Review.* Vehicles, 6(1), 305–351.
4. Yang et al. (2025). *High-Precision Depth Estimation Networks Using Low-Resolution Depth and RGB Image Sensors for Low-Cost Mixed Reality Glasses.* Applied Sciences, 15(11), 6169.
5. NYU Depth V2 Dataset — [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)

---

## Authors

- **Krish Khatri** — 2023A7PS0036U  
- **Karishma Doshi** — 2023A7PS0040U  

BITS Pilani, Dubai Campus — Deep Learning Course (Feb–Jun 2026)
