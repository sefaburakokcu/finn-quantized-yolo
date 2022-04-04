# Low-Precision YOLO on PYNQ with FINN

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Evaluation](#evaluation)


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

**Models**

| Models  | ONNX | Deploy |
| ------------- | ------------- | ------------- |
| 2w4a | [2w4a.onnx](https://1drv.ms/u/s!AoEINH-7w38TknHqzeVmJQplZW2w?e=1nByhb) | [2w4a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TknB8WnaWuPN2Zz7m?e=yt26fl) |
| 3w5a | [3w5a.onnx](https://1drv.ms/u/s!AoEINH-7w38Tkm-HlWVrvBWsNmdL?e=liQFnf) | [3w5a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38Tkm7bsOBtDbwB5fED?e=HLRRHP) |
| 4w2a | [4w2a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkmwXvTQCtZylDW-8?e=OHHCWO) | [4w2a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38Tkm3MW48Nf3x2DXGK?e=o2wy24) |
| 4w4a | [4w4a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkmtJEyY25d04bB-_?e=4woj7w) | [4w4a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkmopP-uPADIHR3Ed?e=8poZTg) |
| 6w4a | [6w4a.onnx](https://1drv.ms/u/s!AoEINH-7w38Tkmjv1Wta4Rtr7L_c?e=0wQeXt) | [6w4a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkmmCQQeV_jJQSCPm?e=mPFI7A) |
| 8w3a | [8w3a.onnx](https://1drv.ms/u/s!AoEINH-7w38TkmZ25kggUNiveWzM?e=hcUMeh) | [8w3a_deploy.zip](https://1drv.ms/u/s!AoEINH-7w38TkmeyAMLvKjqTolG0?e=fhuJ6Q) |

**Inference**

Open _inference.ipynb_ under finn-quantized-yolo/src/deploy/driver in jupyter-notebook.

**Evaluation**

Under _src/deploy/driver/_, run

```bash
python save_inference_results.py --source path-to-widerface-val-folder --outputs ./outputs/
```
