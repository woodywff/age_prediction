'''
10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)

Using 3D-CNN model code of my own
'''
from preprocess import *
from my_tools import *
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import sys


FLAGS = None


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def inference(X, keep_prob, is_training_forBN, trivial=True):
    l2_loss = 0
    with tf.name_scope('l1_conv3d') as scope:
        w = tf.Variable(tf.truncated_normal([5,5,5,1,FLAGS.sizeof_kernel_1], stddev=0.1), name='kernel')
        b = tf.Variable(tf.constant(0.1,shape=[FLAGS.sizeof_kernel_1]),name='b')
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
        w = tf.Variable(tf.truncated_normal([3,3,3,FLAGS.sizeof_kernel_1,32], stddev=0.1), name='kernel')
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
        
def get_loss(predict_batches,label_batches,l2_loss):
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.square(predict_batches - label_batches)) + FLAGS.l2_epsilon * l2_loss
    return cost

def get_accuracy(predicts,labels):
    with tf.name_scope('acc'):
#         acc = tf.contrib.metrics.streaming_pearson_correlation(predicts, tf.cast(labels,tf.float32))[0]
        acc = tf.constant(0)
    return acc



def trainning(loss):
    '''
    The weird things are for Batch Normalization.
    '''
#     train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_op      



def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'arr_raw':tf.FixedLenFeature([],tf.string),
            'label': tf.FixedLenFeature([],tf.float32),
            'id': tf.FixedLenFeature([],tf.int64),
        }
    )
    arr = tf.decode_raw(features['arr_raw'],tf.int64)
    arr = tf.reshape(arr,list(FLAGS.arr_shape))
    label = features['label']
    sub_id = features['id']
    
    return arr,label,sub_id


def get_iterator(for_training=True,num_epochs=1):
    if not num_epochs:
        num_epochs = None
    root_dir = FLAGS.root_dir
    filename = os.path.join(root_dir,('training_data' if for_training else 'test_data')+'.tfrec')
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
#         pdb.set_trace()
        dataset = dataset.map(decode)
        if for_training:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(FLAGS.batch_size)
        if for_training:
            train_iterator = dataset.make_one_shot_iterator()
            val_iterator = dataset.make_one_shot_iterator()
            iterators = [train_iterator,val_iterator]
        else:
            test_iterator = dataset.make_initializable_iterator()
            iterators = [test_iterator]

        
    return iterators

    
def run_training():
    with tf.Graph().as_default():
        num_epochs = FLAGS.num_epochs
        iterators = get_iterator(num_epochs=num_epochs)
        handle = tf.placeholder(tf.string,shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, iterators[0].output_types)
        arr_batch,label_batch,id_batch = iterator.get_next()
        
#         pdb.set_trace()

        X = tf.reshape(arr_batch, [-1]+list(FLAGS.arr_shape)+[1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        is_training_forBN = tf.placeholder(tf.bool, name='is_training_forBN')
        
        predicted_age,l2_loss = inference(X,keep_prob,is_training_forBN,trivial=False)

        loss = get_loss(predicted_age,label_batch,l2_loss)
        acc = get_accuracy(predicted_age,label_batch)
    
        train_op = trainning(loss)
    
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    
        saver = tf.train.Saver()

    
        with tf.Session() as sess:
            losses = []
            acces = []
            steps = []
            
            test_losses = []
            test_acces = []
            test_steps = []
            
#             pdb.set_trace()
            sess.run(init_op)
            train_iterator_handle = sess.run(iterators[0].string_handle())
            val_iterator_handle = sess.run(iterators[1].string_handle())
            
            test_iterator = get_iterator(for_training=False)[0]
            test_iterator_handle = sess.run(test_iterator.string_handle())
            
            print('Let\'s get started to train!')
            
            try:
                step = 0
                min_test_loss = FLAGS.MIN_TEST_LOSS
                while True:
                    start_time = time.time()
                    train_a,train_l,_ = sess.run([arr_batch,label_batch,train_op],
                                                  feed_dict={keep_prob:0.5,
                                                             is_training_forBN:True,
                                                            handle:train_iterator_handle})

                    val_a,val_l,acc_value,loss_value = sess.run([arr_batch,label_batch,acc,loss],
                                                                feed_dict={keep_prob:1.0,
                                                                           is_training_forBN:False,
                                                                          handle:val_iterator_handle})
                    duration = time.time() - start_time
#                     print(type(loss_value))
                    if step % 100 == 0:
                        sess.run(test_iterator.initializer)
                        test_predicted_ages = []
                        test_labels = []
                        try:
                            while True:
                                test_predicted_age,test_label = sess.run([predicted_age,label_batch],
                                                                feed_dict={keep_prob:1.0,
                                                                           is_training_forBN:False,
                                                                          handle:test_iterator_handle})
#                                 pdb.set_trace()
                                test_predicted_ages.append(test_predicted_age)
                                test_labels.append(test_label)
                        except tf.errors.OutOfRangeError:
#                             pdb.set_trace()
                            test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                            test_labels = np.concatenate(tuple(test_labels))
                            test_loss = (get_loss(test_predicted_ages,test_labels,tf.constant(0.0))).eval()
                            test_acc = (get_accuracy(test_predicted_ages,test_labels)).eval()
                            test_losses.append(test_loss)
                            test_acces.append(test_acc)
                            test_steps.append(step)
                            print('Step %d: training_loss = %.2f (%.3f sec)\n (val_loss)test_loss = %.2f' 
                                  %(step,loss_value,duration,test_loss))
#                             print(type(loss_value))
                            
                            if test_loss < min_test_loss:
                                min_test_loss = test_loss
                                print('best shot model: test_loss = %.2f, step = %d' %(min_test_loss,step))
                                if step > 100:
                                    save_path = saver.save(sess, FLAGS.saver_dir)
                                    print('model saved successfully.')
                                    
                    steps.append(step)
                    acces.append(acc_value)
                    losses.append(loss_value)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' %(num_epochs,step))
            

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
            ax1.plot(steps, losses)
            ax1.plot(test_steps,test_losses)
            ax1.set_title('trainning loss')
            ax2.plot(steps, acces)
            ax2.plot(test_steps,test_acces)
            ax2.set_title('trainning acc')
            ax1.grid(True)
            ax2.grid(True)
            plt.savefig('./img/demo_1_1_training.pdf', bbox_inches='tight')
            plt.savefig('./img/demo_1_1_training.png', bbox_inches='tight')
            
            plt_data_path_name = './img/demo_1_1_pltdata_' + time_now()
            if not os.path.exists(plt_data_path_name + '.npy'):
                np.save(plt_data_path_name, np.array([steps,losses,test_steps,test_losses]))
            else:
                print(plt_data_path_name + '.npy exists already.')
            
    return
 
    
def run_test():
    with tf.Graph().as_default():
        handle = tf.placeholder(tf.string,shape=[])
        test_iterator = get_iterator(for_training=False)[0]
        iterator = tf.data.Iterator.from_string_handle(handle, test_iterator.output_types)
        arr_batch,label_batch,id_batch = iterator.get_next()
        X = tf.reshape(arr_batch, [-1]+list(FLAGS.arr_shape)+[1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        is_training_forBN = tf.placeholder(tf.bool, name='is_training_forBN')
        
        predicted_age,l2_loss = inference(X,keep_prob,is_training_forBN)

        loss = get_loss(predicted_age,label_batch,l2_loss)
        acc = get_accuracy(predicted_age,label_batch)
    
        saver = tf.train.Saver()
#         pdb.set_trace()
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.saver_dir)
            print('Model loaded successfully.')
            test_iterator_handle = sess.run(test_iterator.string_handle())
            sess.run(test_iterator.initializer)
            test_predicted_ages = []
            test_labels = []
            try:
                while True:
                    test_predicted_age,test_label = sess.run([predicted_age, label_batch],
                                                    feed_dict={keep_prob:1.0,
                                                               is_training_forBN:False,
                                                              handle:test_iterator_handle})
                    test_predicted_ages.append(test_predicted_age)
                    test_labels.append(test_label)
            except tf.errors.OutOfRangeError:
                test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                test_labels = np.concatenate(tuple(test_labels))
                
                test_loss = (get_loss(test_predicted_ages,test_labels,tf.constant(0.0))).eval()
                test_acc = (get_accuracy(test_predicted_ages,test_labels)).eval()
                
                print('test_loss = %.2f, test_accuracy = %.2f.' 
                                  %(test_loss,test_acc))
            fig = plt.figure()
            plt.title('Test Data')
            plt.xlabel('Chronological Age')
            plt.ylabel('Brain Age (Predicted)')
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            for i in range(len(test_labels)):
                plt.scatter(test_labels[i], test_predicted_ages[i], c = 'blue',s=1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig('./img/demo_1_1_test.pdf', bbox_inches='tight')
            plt.savefig('./img/demo_1_1_test.png', bbox_inches='tight')
            plt.show()
    return




def main(_):
#     pdb.set_trace()
    arr = np.load('./data_npy/mean_npy.npy')
    FLAGS.arr_shape = arr.shape
    
    if not FLAGS.for_test:
        run_training()
    run_test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saver_dir',
                       type=str,
                       default="./log/demo1.1_mine_model.ckpt",
                       help='Directory to save checkpoint.')
    parser.add_argument('--l2_epsilon',
                       type=float,
                       default=1e-6,
                       help='L2 regularization parameter.')
    parser.add_argument('--root_dir',
                       type=str,
                       default='./',
                       help='Where is the tfrecord file.')
    parser.add_argument('--sizeof_kernel_1',
                        type=int,
                        default=16,
                        help='kernel size of 1st hidden layer.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='Batch size.')
#     parser.add_argument(
#       '--learning_rate',
#       type=float,
#       default=0.01,
#       help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--MIN_TEST_LOSS',
                       type=float,
                       default=500.0,
                       help='The minimum test loss value under which the model would be saved.')
    parser.add_argument('--for_test',
                       type=bool,
                       default=False,
                       help='If it is True, the program only runs the test part.')
#     parser.add_argument(
#       '--hidden1',
#       type=int,
#       default=128,
#       help='Number of units in hidden layer 1.')
#     parser.add_argument(
#       '--hidden2',
#       type=int,
#       default=32,
#       help='Number of units in hidden layer 2.')

#     parser.add_argument(
#       '--train_dir',
#       type=str,
#       default='/tmp/data',
#       help='Directory with the training data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)