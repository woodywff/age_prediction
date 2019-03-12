'''
Replication of Ruyi's code.
Dataset: IXI
10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)

# after tar IXI dataset
# rename files
# get phenotypics.csv
# get training.csv test.csv
# get preprocessed .npy files. details of preprocess please refer to the functions.

'''

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
from my_tools import *

# for preprocess_1 resampling
DESIRED_SHAPE_origin=(176,256,256)
DESIRED_SHAPE_resample=(110,130,130)


def get_shuffled(imgs, labels):
    temp = np.array([imgs,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    return image_list,label_list

def gen_phenotypics(xls_path_name):
    '''
    *** This needs to be modified for different programs ***
    This is an once for all thing.
    To generate phenotypics.csv
    
    xls_path_name: excel file with phenotypic information.
    '''
#     pdb.set_trace()
    if os.path.exists('./phenotypics.csv'):
        print('phenotypics.csv exists already.')
        return
    # get id and age from .xls
    phenotypic_table = xlrd.open_workbook(xls_path_name,'rb')
    pt = phenotypic_table.sheets()[0]

    id_list = pt.col_values(0)
    age_list = pt.col_values(1)

    # delete empty items:
    for i in range(len(id_list)-1,0-1,-1):
        if age_list[i] == '':
            del id_list[i]
            del age_list[i]

    # shuffle and save the phenotypic info:
    id_list, age_list = get_shuffled(id_list,age_list)
    data_to_save = pd.DataFrame({'id':id_list,'age':age_list})
    data_to_save.to_csv(os.path.join('./','phenotypics.csv'), index=False, sep=',')
    print('phenotypics.csv created.')
    return

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

def resample(image, pixdim, new_spacing=[1,1,1]):
    '''
    All images are resampled according to the pixel dimension information read from the 
    image header files. 
    This ensures that all images will have the same resolution.
    
    image: ndarray nii_img.get_data()
    pixdim: nii_img.header['pixdim'][1:4]
    
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

def preprocess_1(source_dir,target_dir_origin):
    '''
    preprocess using inner_prerpocess_1()
    step.1: resample
    step.2: crop and padd
    
    '''
    nii_list = os.listdir(source_dir)
     
    pbar = ProgressBar().start()
    n_bar = len(nii_list)
    
    for i,filename in enumerate(nii_list):
        re_result = re.match('^.*\.nii$',filename)
        if re_result:
            target_filename = os.path.join(target_dir_origin,filename.split('.')[0])
            if not os.path.exists(target_filename + '.npy'):
                cropped_npy = inner_preprocess_1(os.path.join(source_dir,filename))
                np.save(target_filename,cropped_npy)
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    return

def inner_preprocess_1(nii_file):
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
#     print('original image shape: ',npy_img.shape)
#     resampled_img = resample(npy_img, pixdim, [2,2,2])
#     print('resampled img shape: ',resampled_img.shape)
    crop_padded_img = crop_pad(npy_img,DESIRED_SHAPE_origin)
#     print('crop and padded img shape: ', crop_padded_img.shape)
    crop_padded_img = np.round(crop_padded_img)
    resampled_img = resample(crop_padded_img, pixdim, [2,2,2])
    crop_padded_img2 = crop_pad(resampled_img,DESIRED_SHAPE_resample)
#     print('crop and padded img shape: ', crop_padded_img.shape)
    crop_padded_img2 = np.round(crop_padded_img2)
    return crop_padded_img2.astype(int)

def preprocess_2(target_dir_origin,target_dir_mean):
    '''
    preprocess
    step.3 subtract mean values
    
    The mean image of all the training data is computed and is subtracted from all training and test data.
    It is worth noting that the test data does not contribute to the mean image. 
    This is because the training data, and only training data, 
    needs to have zero mean for better training performance.
    
    '''
    # get the mean values of all the training data first
    mean_npy = calc_mean()
    
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
                subtracted_npy = origin_npy - mean_npy
                np.save(os.path.join(target_dir_mean,filename.split('.')[0]),subtracted_npy)
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    
    return

def calc_mean():
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
#         str_id = str(int(ixi_id))
#         if ixi_id < 10:
#             str_id = '00' + str_id
#         elif ixi_id > 9 and ixi_id < 100:
#             str_id = '0' + str_id
        npy_filename = ixi_id + '.npy'

        try:
            origin_npy = np.load(os.path.join('./data_npy/origin/',npy_filename))
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        npy_list.append(origin_npy)
        count += 1
    print('Number of training samples: ',count)
    mean_npy = np.mean(npy_list,axis=0)
    mean_npy = np.round(mean_npy)
    mean_npy = mean_npy.astype(int)
    np.save(os.path.join('./data_npy/','mean_npy'),mean_npy)
    return mean_npy
    
    
    
def gen_npy(source_dir,target_dir):
    '''
    To read in .nii.gz files and preprocess, including preprocess_1() and preprocess_2(), 
    then output .npy files in target_dir folder.
    '''
#     pdb.set_trace()
    target_dir_origin = os.path.join(target_dir,'origin')
    target_dir_mean = os.path.join(target_dir,'mean_subtracted')
        
    dirs = [target_dir, target_dir_origin,target_dir_mean]
        
    for path in dirs:
        try:
            os.mkdir(path)
        except FileExistsError:
            print(path,' exists already!')
    
    # preprocess_1: step.1 step.2   
    preprocess_1(source_dir,target_dir_origin)
    # preprocess_2: step.3
    preprocess_2(target_dir_origin,target_dir_mean)
    return


def gen_tfrecord(csv_path_name,npy_dir,tf_filename='tf.tfrecords'):
    '''
    To generate .tfrecord files according to .csv file
    
    csv_path_name: .csv file's path and name
    npy_dir: where are the .npy files
    tf_filename: name of .tfrecords file under ./
    '''
#     pdb.set_trace()
    
    tf_path_name = os.path.join('./',tf_filename)

#     tf_path = os.path.join(data_root,feature_name, 'train' if for_training else 'test')
#     tf_filename = os.path.join(tf_path, '.tfrecords')
    if os.path.exists(tf_path_name):
        print(tf_path_name, 'exists already.')
        return
    print('Writing', tf_path_name)
    with tf.python_io.TFRecordWriter(tf_path_name) as writer:
#         df_name = os.path.join(tf_path,'info.csv')
        info_df = pd.read_csv(csv_path_name, sep=',',header=0)
        id_list = list(info_df['id'])
        age_list = list(info_df['age'])
        n = len(id_list)
        pbar = ProgressBar().start()
        for i,ixi_id in enumerate(id_list):
#             str_id = str(int(ixi_id))
#             if ixi_id < 10:
#                 str_id = '00' + str_id
#             elif ixi_id > 9 and ixi_id < 100:
#                 str_id = '0' + str_id
            npy_filename = ixi_id + '.npy'
            npy_path_filename = os.path.join(npy_dir,npy_filename)
            try:
                arr_npy = np.load(npy_path_filename)
            except FileNotFoundError:
                print('No such file: ',npy_filename)
                continue
                
#             pdb.set_trace()
            label = round(age_list[i],2)
            arr = arr_npy.astype(np.float32)
            arr_raw = arr.tostring()
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'arr_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw])),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
                    }
                )
            )
            writer.write(example.SerializeToString())
            pbar.update(int(i*100/(n-1)))
        pbar.finish()
    return

def preprocess_main():
    print_sep('preprocessing starts')
    # get phenotypics.csv
    gen_phenotypics('/media/woody/Elements/Steve_age_data/participants.xlsx')
    # get training.csv test.csv
    gen_training_test_csv()
    # get preprocessed .npy files
    source_dir = '/media/woody/Elements/Steve_age_data/ANAT'
    target_dir = './data_npy'
    gen_npy(source_dir,target_dir)
    # get .tfrecords ready
    gen_tfrecord('./training.csv',npy_dir='./data_npy/mean_subtracted/',tf_filename='training.tfrecords')
    gen_tfrecord('./test.csv',npy_dir='./data_npy/mean_subtracted/',tf_filename='test.tfrecords')
    
    my_mkdir('./img')
    my_mkdir('./log')
    print_sep('preprocessing ends')
    return
    


if __name__ == '__main__':
    preprocess_main()
    
    