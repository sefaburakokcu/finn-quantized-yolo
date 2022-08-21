# LPYOLO: Low Precision YOLO for Face Detection on FPGA

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

This repo contains evaluation and deploy scripts for official [LPYOLO: Low Precision YOLO for Face Detection on FPGA](https://arxiv.org/abs/2207.10482) paper.
Models are trained by [Brevitas](https://github.com/Xilinx/brevitas) which is a PyTorch research library for quantization-aware training (QAT) and exported to [ONNX]([https://onnx.ai). [FINN](https://github.com/Xilinx/finn) which is an experimental framework from Xilinx Research Labs to explore deep neural network inference on FPGAs is used for deploying models on a [PYNQ-Z2](http://www.pynq.io/board.html) board.


## Requirements

* Finn == 0.7
* Pytorch >= 1.8.1

## Installation

First, download [Pytorch armv7](https://github.com/KumaTea/pytorch-arm/releases/download/v1.8.1/torch-1.8.1-cp38-cp38-linux_armv7l.whl) and
[Torchvision armv7](https://github.com/KumaTea/pytorch-arm/releases/download/v1.8.1/torchvision-0.9.1-cp38-cp38-linux_armv7l.whl).

Then, run
```bash
pip install torch-1.8.1-cp38-cp38-linux_armv7l.whl
pip install torchvision-0.9.1-cp38-cp38-linux_armv7l.whl
```
on PYNQ-Z2 board.

Also, install bitstream
```bash
pip install bitstream
```

## Usage

First, connect a PYNQ-Z2 board and open a terminal.Then, clone the project:

```bash
git clone git@github.com:sefaburakokcu/finn-quantized-yolo.git
```
Then, download one of the deploy.zip file from the table below and extract. Copy _finn-accel.bit_, _finn-accel.hwh_ and _scale.npy_ into _src/deploy/bitfile/_.

**Model Definition**

Definition of LPYOLO architecture is given below.

![Model Definition](inputs/docs/lp_yolo_architecture.png)

**Pretrained Models**

| Models  | ONNX | Deploy |
| ------------- | ------------- | ------------- |
| 2W4A | [2w4a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkwFlg_07Yhh76hPE?e=Ma1IWY) | [2w4a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38Tkwfd6XLDIm8k-EbH?e=Xww5yf) |
| 3W5A | [3w5a.onnx](https://1drv.ms/u/s!AoEINH-7w38Tkwau-ObdlbiQtmGV?e=mptSyU) | [3w5a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkwkBL6jLCCvWHIDN?e=UWkWm3) |
| 4W2A | [4w2a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkwPBWHfqFpwRWxF9?e=SFYNHG) | [4w2a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkwijP2bavoCA64AF?e=E1JR1j) |
| 4W4A | [4w4a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkwLx0_GK3lpJIZ9T?e=cKusXD) | [4w4a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkwrR9L7i4bVyI6iO?e=fSSZbW) |
| 6W4A | [6w4a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkwRrcGTD_POag352?e=yDTjj0) | [6w4a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38Tkwu76ecp2F2UvGww?e=kilEft) |
| 8W3A | [8w3a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkwV-1W_SFhbQ5-QZ?e=VRsJ98) | [8w3a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkwxIVbTOayliPFV1?e=hV9ucA) |

Note: All models have 8 bits input and 8 bits output precisions. xWyA indicates x bits for weights and y bits precision for activations.

**Training**

Please use the following [quantized-yolov5](https://github.com/sefaburakokcu/quantized-yolov5) repository for training quantized models.

**Inference**

Open _inference.ipynb_ for inference on images, _video_demo.ipynb_ for inference on a video under finn-quantized-yolo/src/deploy/ in jupyter-notebook.

