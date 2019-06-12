# age_prediction
## Introduction
This project tried to use 3D-CNN to predict brain age from structural and functional MRI images.

There are 4 different datasets: ABIDEI, ABIDEII, IXI, and NKI. Each of them has one specific folder named after like `age_demo_abide1` inside which there is one `phenotypics.csv` saving the information of subjects' id and age, and one `preprocess.py` specifically used for the preprocessing of that dataset. 

For all the 4 datasets we did the prediction based on the raw T1 structural images.

For NKI dataset we also tried the functional modalities including ALFF, FALFF and REHO. Meanwhile, we provided one model combined by the pretrained networks of ALFF and T1.

`dev_models` has the model files.

`dev_tools` provides functions used throughout this project.

## How to run the demo
This project had been developed under the environment of
```
Ubuntu16.04 + python3.5 + tensorflow-gpu1.11.0
```
Because the downloaded datasets were stored on my mobile HDD, you may encounter some path name like `/media/woody/Elements`, that's my disk's name, you could change it into your pathway.

Anyway, you could run it like this (in jupyter notebook):
```
# Mostly because of the data storing structure differs from one dataset to another, 
# each dataset has its own preprocessing function.  

import preprocess
preprocess.preprocess_main()

!python ../../dev_models/model_woody.py

```
The `Script.ipynb` gives out a brief show of some illustration functions you may wanna invoke when checking the training and test results.
