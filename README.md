# Low-Precision YOLO on PYNQ with FINN

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)


## Introduction

This repo contains evaluation, export and deploy scripts of yolo models.
Models are trained by [Brevitas](https://github.com/Xilinx/brevitas) which is a PyTorch research library for quantization-aware training (QAT) and exported to [ONNX]([https://onnx.ai). FINN](https://github.com/Xilinx/finn) which is an experimental framework from Xilinx Research Labs to explore deep neural network inference on FPGAs is used for deploying models on a [PYNQ-Z2](http://www.pynq.io/board.html) board.


## Requirements

* Finn == 0.6
* Onnx

## Usage

First, connect a PYNQ-Z2 board and open a terminal.Then, clone the project:

```bash
git clone git@github.com:sefaburakokcu/finn-quantized-yolo.git
```

**Inference**

Open _inference.ipynb_ under finn-quantized-yolo/src/deploy in jupyter-notebook.

