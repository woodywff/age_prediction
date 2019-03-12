import xlrd
import numpy as np
import pandas as pd
import os
import re
from progressbar import *
import nibabel as nib
import pdb
import scipy.ndimage
import matplotlib.pyplot as plt
import time

def print2d(npy_img,save=False,save_name='./test.jpg'):
    '''
    plot 2d mri images in Sagittal, Coronal and Axial dimension.
    img: 3d ndarray
    '''
    dim = npy_img.shape
    print('Dimension: ',npy_img.shape)
    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15,5))

#     img = npy_img[:,:,round(dim[2]/2)]
    img = npy_img[87,:,:]
    ax1.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax1.set_title('Sagittal',fontsize=15)
#     ax1.imshow(img, cmap=plt.cm.gray)
    ax1.axis('off')
#     img = npy_img[:,round(dim[1]/2),:]
    img = npy_img[:,123,:]
    ax2.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax2.set_title('Coronal',fontsize=15)
    ax2.axis('off')
#     img = npy_img[round(dim[0]/2),:,:]
    img = npy_img[:,:,154]
    ax3.imshow(np.rot90(img), cmap=plt.cm.gray)
    ax3.set_title('Axial',fontsize=15)
#     ax3.imshow(img, cmap=plt.cm.gray)
    ax3.axis('off')
    # plt.subplot(131); plt.imshow(np.rot90(img), cmap=plt.cm.gray)
    # img = npy_img[:,65,:]
    # plt.subplot(132); plt.imshow(img, cmap=plt.cm.gray)
    # img = npy_img[65,:,:]
    # plt.subplot(133); plt.imshow(np.rot90(img,2), cmap=plt.cm.gray)
    if save:
        plt.savefig(save_name)
    return

def printimg(filename,size=10):
    f, (ax1) = plt.subplots(1, 1, figsize=(size,size))
    npy = plt.imread(filename)
    ax1.imshow(npy)
    ax1.axis('off')
    return

def print_sep(something='-'):
    print('----------------------------------------',something,'----------------------------------------')
    return

# 3D rotatation
def rot_clockwise(arr,n=1):
    return np.rot90(arr,n,(0,2))
def rot_anticlockwise(arr,n=1):
    return np.rot90(arr,n,(2,0))

def time_now():
    return time.strftime('%Y.%m.%d.%H:%M:%S',time.localtime(time.time()))

def my_mkdir(path_name):
    try:
        os.mkdir(path_name)
    except FileExistsError:
        print(path_name,' exists already!')
    return