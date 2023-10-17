# Eigenpatches - Adversarial Patches from Principal Components
~ **Work in Progress** ~

## Preparations
- Clone this repository.
- Initialize the yolov7 submodule and apply the `advAttack.patch`.
- In addition to the requirements of yolov7, install `scikit-learn`.
- Download the INRIA Person dataset and adjust the paths in `train.py:57` and `evaluate.py:54` accordingly. 

## Patch Training
Check `train.py` and adjust the config dict as you wish.

## Evaluate
Check `evaluate.py:6` and change `EVALUATION_TYPE` to either `nelems` or `ndims`.

## Sampler
Run or import `sampler.py` directly. The `outputs/*.pth` files can be used when directly running the file. 

