# import sys
# sys.path.append("..")
# from dev_tools.model_tools import *
# from dev_tools.preprocess_tools import *
import numpy as np
import tensorflow as tf
import pandas as pd

n_classes = 1

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')
def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
def inference(x,dropout_rate,trivial=True):
    initializer = tf.contrib.layers.xavier_initializer()
    weights = {'W_conv1':tf.Variable(initializer([3,3,3,1,8]), name="W_conv1"),
               'W_conv2':tf.Variable(initializer([3,3,3,8,16]), name="W_conv2"),
               'W_conv3':tf.Variable(initializer([3,3,3,16,32]), name="W_conv3"),
               'W_conv4':tf.Variable(initializer([3,3,3,32,64]), name="W_conv4"),
               'W_conv5':tf.Variable(initializer([3,3,3,64,128]), name="W_conv5"),
#                'W_fc':tf.Variable(initializer([temp_shape,1024]), name="W_fc"),
               'W_out':tf.Variable(initializer([1024, n_classes]), name="W_out")}

    biases = {'b_conv1':tf.Variable(initializer([8]), name="b_conv1"),
               'b_conv2':tf.Variable(initializer([16]), name="b_conv2"),
              'b_conv3':tf.Variable(initializer([32]), name="b_conv3"),
              'b_conv4':tf.Variable(initializer([64]), name="b_conv4"),
              'b_conv5':tf.Variable(initializer([128]), name="b_conv5"),
               'b_fc':tf.Variable(initializer([1024]), name="b_fc"),
               'b_out':tf.Variable(initializer([n_classes]), name="b_out")}

#     x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = tf.nn.dropout(conv1, dropout_rate)
    conv1 = maxpool3d(conv1)
    if trivial:
        print_activations(conv1)
    
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = tf.nn.dropout(conv2, dropout_rate)
    conv2 = maxpool3d(conv2)
    if trivial:
        print_activations(conv2)

    conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = tf.nn.dropout(conv3, dropout_rate)
    conv3 = maxpool3d(conv3)
    if trivial:
        print_activations(conv3)

    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = tf.nn.dropout(conv4, dropout_rate)
    conv4 = maxpool3d(conv4)
    if trivial:
        print_activations(conv4)
        
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5']) + biases['b_conv5'])
    conv5 = tf.nn.dropout(conv5, dropout_rate)
    conv5 = maxpool3d(conv5)
    if trivial:
        print_activations(conv5)
    
    max_pool_shape = conv5.get_shape().as_list()
    temp_shape = 1
    for i in max_pool_shape[1:]:
        temp_shape *= i
    fc_input = tf.reshape(conv5, [-1, temp_shape])
    
    w = tf.Variable(tf.truncated_normal([temp_shape,1024],stddev=0.1),name='W_fc')
    fc = tf.matmul(fc_input, w)+biases['b_fc']
    fc = tf.nn.dropout(fc, dropout_rate)
    if trivial:
        print_activations(fc)

    output = tf.add(tf.matmul(fc, weights['W_out']),biases['b_out'],name="output")
    if trivial:
        print_activations(output)    

#     return output[0]
    return output

def test_single_subject(phe_index,model_path='./log/model_mse_zach.ckpt'):
    '''
    model_path: './log/model_mse_xxx.ckpt' or './log/model_person_xxx.ckpt'
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
        
        predicted_age = inference(X,keep_prob,trivial=False)

        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            print('Model loaded successfully.')
                
            p_age = sess.run(predicted_age,feed_dict={keep_prob:1.0,
                                              tf_arr:arr,
                                              })
            print('Subject: ',sub_id,', chronological age is ',
                  label,', predicted age is ',p_age,'.')
    return
