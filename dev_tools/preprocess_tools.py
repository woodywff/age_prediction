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
import tensorflow as tf
from dev_tools.my_tools import *
import tarfile
import shutil



def calc_mean():
    '''
    this fun varies in some datasets like IXI, in IXI we use calc_mean_ixi()
    This code is self-content, which means it sucks and could be polished.
    '''
    if os.path.exists('./data_npy/mean_npy.npy'):
        print('./data_npy/mean_npy.npy exists already.')
        mean_npy = np.load('./data_npy/mean_npy.npy')
        return mean_npy
    
    train_df = pd.read_csv('./training.csv', sep=',',header=0)
    id_list = train_df['id']
    

#     pdb.set_trace()
    count = 0
    npy_list = []
    for sub_id in id_list:
        npy_filename = str(int(sub_id)) + '.npy'

        try:
            origin_npy = np.load(os.path.join('./data_npy/origin/',npy_filename))
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        npy_list.append(origin_npy)
        count += 1
    print('Number of training samples: ',count)
    mean_npy = np.mean(npy_list,axis=0)
#     mean_npy = np.round(mean_npy)
#     mean_npy = mean_npy.astype(int)
    np.save(os.path.join('./data_npy/','mean_npy'),mean_npy)
    return mean_npy

def calc_std():
    '''
    This code is self-content, which means it sucks and could be polished.
    '''
    if os.path.exists('./data_npy/std_npy.npy'):
        print('./data_npy/std_npy.npy exists already.')
        std_npy = np.load('./data_npy/std_npy.npy')
        return std_npy
    
    train_df = pd.read_csv('./training.csv', sep=',',header=0)
    id_list = train_df['id']
    

#     pdb.set_trace()
    count = 0
    npy_list = []
    for sub_id in id_list:
        npy_filename = str(int(sub_id)) + '.npy'

        try:
            origin_npy = np.load(os.path.join('./data_npy/origin/',npy_filename))
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        npy_list.append(origin_npy)
        count += 1
    print('Number of training samples: ',count)
    std_npy = np.std(npy_list,axis=0)
#     mean_npy = np.round(mean_npy)
#     mean_npy = mean_npy.astype(int)
    np.save(os.path.join('./data_npy/','std_npy'),std_npy)
    return std_npy


def calc_mean_ixi():
    '''
    
    
    This code is self-content, which means it sucks and could be polished.
    '''
    if os.path.exists('./data_npy/mean_npy.npy'):
        print('./data_npy/mean_npy.npy exists already.')
        mean_npy = np.load('./data_npy/mean_npy.npy')
        return mean_npy
    
    train_df = pd.read_csv('./training.csv', sep=',',header=0)
    id_list = train_df['id']

#     pdb.set_trace()
    count = 0
    npy_list = []
    for ixi_id in id_list:
        str_id = str(int(ixi_id))
        if ixi_id < 10:
            str_id = '00' + str_id
        elif ixi_id > 9 and ixi_id < 100:
            str_id = '0' + str_id
        npy_filename = 'IXI' + str_id + '.npy'

        try:
            origin_npy = np.load(os.path.join('./data_npy/origin/',npy_filename))
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        npy_list.append(origin_npy)
        count += 1
    print('Number of training samples: ',count)
    mean_npy = np.mean(npy_list,axis=0)
#     mean_npy = np.round(mean_npy)
#     mean_npy = mean_npy.astype(int)
    np.save(os.path.join('./data_npy/','mean_npy'),mean_npy)
    return mean_npy

def calc_std_ixi():
    '''
    
    
    This code is self-content, which means it sucks and could be polished.
    '''
    if os.path.exists('./data_npy/std_npy.npy'):
        print('./data_npy/std_npy.npy exists already.')
        std_npy = np.load('./data_npy/std_npy.npy')
        return std_npy
    
    train_df = pd.read_csv('./training.csv', sep=',',header=0)
    id_list = train_df['id']

#     pdb.set_trace()
    count = 0
    npy_list = []
    for ixi_id in id_list:
        str_id = str(int(ixi_id))
        if ixi_id < 10:
            str_id = '00' + str_id
        elif ixi_id > 9 and ixi_id < 100:
            str_id = '0' + str_id
        npy_filename = 'IXI' + str_id + '.npy'

        try:
            origin_npy = np.load(os.path.join('./data_npy/origin/',npy_filename))
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        npy_list.append(origin_npy)
        count += 1
    print('Number of training samples: ',count)
    std_npy = np.std(npy_list,axis=0)
#     mean_npy = np.round(mean_npy)
#     mean_npy = mean_npy.astype(int)
    np.save(os.path.join('./data_npy/','std_npy'),std_npy)
    return std_npy

def resample(image, pixdim, new_spacing=[4,4,4]):
    '''
    (This func is copied from Zach's code)
    All images are resampled according to the pixel dimension information read from the 
    image header files. 
    This ensures that all images will have the same resolution.
    
    image: ndarray nii_img.get_data()
    pixdim: nii_img.header['pixdim'][1:4]
    new_spaceing: new pixdim, we could take is as how many points each new pixel represents.
    
    return: ndarray
    '''
    spacing = pixdim

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
#     new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image

def crop_pad(image,desired_shape):
    '''
    To crop or pad images to the same shape
    this function may cause problem when cropping.
    
    image: ndarray
    desired_shape: (130,130,110) like tuple
    
    return: ndarray
    '''
    X_margin_0 = int((desired_shape[0]-image.shape[0])/2)
    Y_margin_0 = int((desired_shape[1]-image.shape[1])/2)
    Z_margin_0 = int((desired_shape[2]-image.shape[2])/2)
    
    X_margin_1 = desired_shape[0]-image.shape[0]-X_margin_0
    Y_margin_1 = desired_shape[1]-image.shape[1]-Y_margin_0
    Z_margin_1 = desired_shape[2]-image.shape[2]-Z_margin_0
    
    npad = ((X_margin_0,X_margin_1), 
            (Y_margin_0,Y_margin_1), 
            (Z_margin_0,Z_margin_1))
    crop_padded_img = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    return crop_padded_img

def crop_center(img,cropx,cropy, cropz):
    x,y,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[startx:startx+cropx,starty:starty+cropy, startz:startz+cropz]

def crop_pad_abide(image,desired_shape):
    cropx = image.shape[0]
    cropy = image.shape[1]
    cropz = image.shape[2]
    if desired_shape[0] < image.shape[0]:
        cropx = desired_shape[0]
    if desired_shape[1] < image.shape[1]:
        cropy = desired_shape[1]
    if desired_shape[2] < image.shape[2]:
        cropz = desired_shape[2]

    cropped_img = crop_center(image, cropx, cropy, cropz)

    X_before = int((desired_shape[0]-cropped_img.shape[0])/2)
    Y_before = int((desired_shape[1]-cropped_img.shape[1])/2)
    Z_before = int((desired_shape[2]-cropped_img.shape[2])/2)

    npad = ((X_before, desired_shape[0]-cropped_img.shape[0]-X_before), 
            (Y_before, desired_shape[1]-cropped_img.shape[1]-Y_before), 
            (Z_before, desired_shape[2]-cropped_img.shape[2]-Z_before))
    crop_padded_img = np.pad(cropped_img, pad_width=npad, mode='constant', constant_values=0)
    return crop_padded_img