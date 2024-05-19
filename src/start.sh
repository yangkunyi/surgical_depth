#!/bin/bash

cd /bd_byta6000i0/users/surgical_depth/surgical_depth/src

/bd_byta6000i0/users/surgical_depth/miniforge3/bin/conda run -n sd --no-capture-output python train.py trainer=overfit model=overfit callbacks=overfit
