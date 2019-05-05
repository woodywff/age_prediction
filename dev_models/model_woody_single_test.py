# import sys
# sys.path.append("..")
# from dev_tools.model_tools import *
# from dev_tools.preprocess_tools import *
import numpy as np
import tensorflow as tf
import pandas as pd

def inference(X, keep_prob, is_training_forBN, trivial=True):
    l2_loss = 0
    with tf.name_scope('l1_conv3d') as scope:
        w = tf.Variable(tf.truncated_normal([5,5,5,1,16], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1,shape=[16]),name='b')
        temp_output = tf.nn.bias_add(tf.nn.conv3d(X,w,strides=[1,1,1,1,1],\
                                                        padding='SAME',name='conv3d'),b)
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        conv3d = tf.nn.relu(temp_output)
        
        max_pool = tf.nn.max_pool3d(conv3d,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],\
                                   padding='SAME',name='max_pool3d')
        
#         l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(conv3d)
            print_activations(max_pool)
    with tf.name_scope('l2_conv3d') as scope:
        w = tf.Variable(tf.truncated_normal([3,3,3,16,32], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1,shape=[32]),name='b')
        temp_output = tf.nn.bias_add(tf.nn.conv3d(max_pool,w,strides=[1,1,1,1,1],\
                                                        padding='SAME',name='conv3d'),b)
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        conv3d = tf.nn.relu(temp_output)
        
        max_pool = tf.nn.max_pool3d(conv3d,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],\
                                   padding='SAME',name='max_pool3d')
        
#         l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(conv3d)
            print_activations(max_pool)
    with tf.name_scope('l3_conv3d') as scope:
        w = tf.Variable(tf.truncated_normal([3,3,3,32,64], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1,shape=[64]),name='b')
        temp_output = tf.nn.bias_add(tf.nn.conv3d(max_pool,w,strides=[1,1,1,1,1],\
                                                        padding='SAME',name='conv3d'),b)
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        conv3d = tf.nn.relu(temp_output)
        
        max_pool = tf.nn.max_pool3d(conv3d,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],\
                                   padding='SAME',name='max_pool3d')
        
#         l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(conv3d)
            print_activations(max_pool)
    with tf.name_scope('l4_conv3d') as scope:
        w = tf.Variable(tf.truncated_normal([3,3,3,64,64], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1,shape=[64]),name='b')
        temp_output = tf.nn.bias_add(tf.nn.conv3d(max_pool,w,strides=[1,1,1,1,1],\
                                                        padding='SAME',name='conv3d'),b)
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        conv3d = tf.nn.relu(temp_output)
        
        max_pool = tf.nn.max_pool3d(conv3d,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],\
                                   padding='SAME',name='max_pool3d')
        
#         l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(conv3d)
            print_activations(max_pool)
    with tf.name_scope('l5_conv3d') as scope:  # temp
        w = tf.Variable(tf.truncated_normal([3,3,3,64,64], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1,shape=[64]),name='b')
        temp_output = tf.nn.bias_add(tf.nn.conv3d(max_pool,w,strides=[1,1,1,1,1],\
                                                        padding='SAME',name='conv3d'),b)
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        conv3d = tf.nn.relu(temp_output)
        
        max_pool = tf.nn.max_pool3d(conv3d,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],\
                                   padding='SAME',name='max_pool3d')
        
#         l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(conv3d)
            print_activations(max_pool)
    with tf.name_scope('l6_fc') as scope:
        max_pool_shape = max_pool.get_shape().as_list()
        temp_shape = 1
        for i in max_pool_shape[1:]:
            temp_shape *= i
        fc_input = tf.reshape(max_pool, [-1, temp_shape])
        w = tf.Variable(tf.truncated_normal([temp_shape,512],stddev=0.1),name='w')
        b = tf.Variable(tf.constant(0.1,shape=[512]),name='b')
        temp_output = tf.matmul(fc_input,w) + b
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        fc_out = tf.nn.relu(temp_output, name='fc_out1')
        dropout = tf.nn.dropout(fc_out,keep_prob=keep_prob, name='dropout1')
        
        l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(fc_out)
            print_activations(dropout)
    with tf.name_scope('l7_fc') as scope:
        w = tf.Variable(tf.truncated_normal([512,128],stddev=0.1),name='w')
        b = tf.Variable(tf.constant(0.1,shape=[128]),name='b')
        temp_output = tf.matmul(fc_out,w) + b
        temp_output = tf.layers.batch_normalization(temp_output,training=is_training_forBN)
        fc_out = tf.nn.relu(temp_output, name='fc_out2')
        dropout = tf.nn.dropout(fc_out,keep_prob=keep_prob, name='dropout2')
        
        l2_loss += tf.nn.l2_loss(w)
        
        if trivial:
            print_activations(fc_out)
            print_activations(dropout)
    with tf.name_scope('l8_fc') as scope:
        w = tf.Variable(tf.truncated_normal([128,1],stddev=0.1),name='w')
        b = tf.Variable(tf.constant(0.1,shape=[1]),name='b')
        
        final_output = tf.add(tf.matmul(dropout,w), b, name='final_output')
        
        l2_loss += tf.nn.l2_loss(w)

        if trivial:
            print_activations(final_output)
    
    return final_output, l2_loss

def test_single_subject(phe_index,model_path='./log/model_mse_woody.ckpt'):
    '''
    '''
#     pdb.set_trace()
    with tf.Graph().as_default():
        phe = pd.read_csv('./phenotypics.csv', sep=',',header=0)
        sub_id = phe['id'][phe_index]
        arr = np.load('./data_npy/mean_subtracted/'+str(int(sub_id))+'.npy')
        arr = arr.astype(np.float32)
        arr_shape = arr.shape
        label = phe['age'][phe_index]
        
        tf_arr = tf.placeholder(tf.float32,shape=arr_shape)
#         tf_label = tf.placeholder(tf.float32)
        X = tf.reshape(tf_arr, [-1]+list(arr_shape)+[1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        is_training_forBN = tf.placeholder(tf.bool, name='is_training_forBN')
        
        predicted_age,l2_loss = inference(X,keep_prob,is_training_forBN,trivial=False)

        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            print('Model loaded successfully.')
                
            p_age = sess.run(predicted_age,feed_dict={keep_prob:1.0,
                                              is_training_forBN:False,
                                              tf_arr:arr,
                                              })
            print('Subject: ',sub_id,', chronological age is ',
                  label,', predicted age is ',p_age,'.')
    return
