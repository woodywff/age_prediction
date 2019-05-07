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


def gen_training_test_csv():
    '''
    This is an once for all thing.
    to generate training.csv and test.csv
    '''
    if os.path.exists('./training.csv') and os.path.exists('./test.csv'):
        print('training.csv and test.csv exist already.')
        return
    phenotypics = pd.read_csv('./phenotypics.csv', sep=',',header=0)
    mid_point = round(0.9 * len(phenotypics))
    training_df = phenotypics[:mid_point]
    test_df = phenotypics[mid_point:]
    training_df.to_csv(os.path.join('./','training.csv'), index=False, sep=',')
    print('training.csv created.')
    test_df.to_csv(os.path.join('./','test.csv'), index=False, sep=',')
    print('test.csv created.')
    return  


def preprocess_2(target_dir_origin,target_dir_mean,std_version=False,int_version=False):
    '''
    preprocess
    step.3 subtract mean values
    
    The mean image of all the training data is computed and is subtracted from all training and test data.
    It is worth noting that the test data does not contribute to the mean image. 
    This is because the training data, and only training data, 
    needs to have zero mean for better training performance.
    
    std_version: whether to divide the standard variation
    int_version: whether to round() and astype(int) the mean values
    '''
    # get the mean values of all the training data first
    mean_npy = calc_mean(int_version=int_version)
    if std_version:
        std_npy = calc_std()
    
    npy_list = os.listdir(target_dir_origin)
    
    pbar = ProgressBar().start()
    n_bar = len(npy_list)
    
    for i,filename in enumerate(npy_list):
        re_result = re.match('^.*\.npy$',filename)
        if re_result:
            target_filename = os.path.join(target_dir_mean,filename)
            if not os.path.exists(target_filename):
                try:
                    origin_npy = np.load(os.path.join(target_dir_origin,filename))
                except FileNotFoundError:
                    print('No such file: ',npy_filename)
                    continue
                if std_version:
                    subtracted_npy = (origin_npy - mean_npy)/std_npy
                else:
                    subtracted_npy = origin_npy - mean_npy
                
                np.save(os.path.join(target_dir_mean,filename.split('.')[0]),subtracted_npy)
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    
    return


def inner_preprocess_1(nii_file,new_shape=(67,67,67),abide=False,rot_ixi=False):
    '''
    preprocess
    step.1: resample
    step.2: crop and padd
    
    nii_file: absolute path of .nii.gz file
    
    return: ndarray
    '''
    
    nii_img = nib.load(nii_file)
    header = nii_img.header
    pixdim = np.round(header['pixdim'][1:4],4)
    npy_img = nii_img.get_data()
    resampled_img = resample(npy_img, pixdim)
    
    if rot_ixi:
        rotated_img = rot_ixi2abide(resampled_img)
    
    if abide:
        crop_padded_img = crop_pad_abide(resampled_img,new_shape)
    else:
        crop_padded_img = crop_pad(resampled_img,new_shape)
    crop_padded_img = np.round(crop_padded_img)
    return crop_padded_img.astype(int)


def gen_tfrecord(csv_path_name,npy_dir,tf_filename='tf.tfrecords'):
    '''
    To generate .tfrecord files according to .csv file
    
    csv_path_name: .csv file's path and name
    npy_dir: where are the .npy files
    tf_filename: name of .tfrecords file under ./
    '''
#     pdb.set_trace()
    
    tf_path_name = os.path.join('./',tf_filename)

    if os.path.exists(tf_path_name):
        print(tf_path_name, 'exists already.')
        return
    print('Writing', tf_path_name)
    with tf.python_io.TFRecordWriter(tf_path_name) as writer:
        info_df = pd.read_csv(csv_path_name, sep=',',header=0)
        id_list = list(info_df['id'])
        age_list = list(info_df['age'])
        n = len(id_list)
        pbar = ProgressBar().start()
        for i,sub_id in enumerate(id_list):
            npy_filename = str(int(sub_id)) + '.npy'
            npy_path_filename = os.path.join(npy_dir,npy_filename)
            try:
                arr_npy = np.load(npy_path_filename)
            except FileNotFoundError:
                print('No such file: ',npy_filename)
                continue
                
#             pdb.set_trace()
            label = round(age_list[i],2)
            arr_npy = arr_npy.astype(np.float32)
            arr_raw = arr_npy.tostring()
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'arr_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw])),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(sub_id)])),
                    }
                )
            )
            writer.write(example.SerializeToString())
            pbar.update(int(i*100/(n-1)))
        pbar.finish()
    return

def calc_mean(int_version=False):
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
    if int_version:
        mean_npy = np.round(mean_npy)
        mean_npy = mean_npy.astype(int)
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