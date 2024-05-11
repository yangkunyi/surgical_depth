#!/bin/bash

cd /bd_byta6000i0/users/surgical_depth/surgical_depth/src

/bd_byta6000i0/users/surgical_depth/micromamba run -p /bd_byta6000i0/users/surgical_depth/.micromamba_data/envs/sd python train.py trainer=overfit model=overfit callbacks=overfit
