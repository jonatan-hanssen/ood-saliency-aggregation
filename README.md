# Repository for paper "XAI Saliency Maps Can Be Used for OOD Detection"

This repository contains the implementation of the SalAgg+MLS framework from the paper "XAI Saliency Maps Can Be Used for OOD Detection".

# Installation

Requires ```uv``` and a C++ compiler (for ```libmr```, an OpenOOD dependency), for example ```gcc```. After this, simply run ```uv run main.py``` and AUROC scores will be generated for the given combination of benchmark, XAI method and aggregator function.

```
usage: main.py [-h] [--dataset [{cifar10,cifar100,imagenet200,imagenet}]] [--saliency_generator [{gbp,gradcam,lime,occlusion,integratedgradients}]]
               [--aggregator [{mean,median,norm,range,max,q3,cv,rmd,qcd}]] [--batch_size BATCH_SIZE] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --dataset, -d [{cifar10,cifar100,imagenet200,imagenet}]
  --saliency_generator, -s [{gbp,gradcam,lime,occlusion,integratedgradients}]
  --aggregator, -a [{mean,median,norm,range,max,q3,cv,rmd,qcd}]
  --batch_size, -b BATCH_SIZE
  --device DEVICE
```

