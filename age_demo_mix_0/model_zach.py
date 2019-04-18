'''
Replication of Ruyi's code.

10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)

Using ruyi's model code
'''

from preprocess import *
import argparse
import sys


FLAGS = None
n_classes = 1

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')
def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
def convolutional_neural_network(x,dropout_rate,trivial=True):
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


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

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
    arr = tf.cast(arr,tf.float32) # to be compliance with the restriction of TypeError: 
                                  # Value passed to parameter 'input' has DataType int64 not in list of allowed values: 
                                  # float16, bfloat16, float32, float64
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

def training_figure_iterator():
    dataset = tf.data.TFRecordDataset('./training_data.tfrec')
    dataset = dataset.map(decode)
    dataset = dataset.batch(FLAGS.batch_size)
    return [dataset.make_initializable_iterator()]
    
    

def get_loss(predict_batches,label_batches):
    with tf.name_scope('RMSE'):
        loss = tf.reduce_mean(tf.square(predict_batches - label_batches)) 
    return loss

    
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
        
        prediction = convolutional_neural_network(X,keep_prob)
#         pdb.set_trace()
        loss = get_loss(prediction,label_batch)
        train_op = tf.train.AdamOptimizer(learning_rate=7e-5).minimize(loss)        
    
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
    
        saver = tf.train.Saver()
    
        with tf.Session() as sess:
            losses = []
            steps = []
            
            test_losses = []
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
                    sess.run(train_op,
                              feed_dict={keep_prob:0.8,
                                        handle:train_iterator_handle})

                    loss_value = sess.run(loss,
                                        feed_dict={keep_prob:1.0,
                                                  handle:val_iterator_handle})
#                     print(type(loss_value))
                    duration = time.time() - start_time

                    if step % 100 == 0:
                        sess.run(test_iterator.initializer)
                        test_predicted_ages = []
                        test_labels = []
                        try:
                            while True:
                                test_predicted_age,test_label = sess.run([prediction,label_batch],
                                                                feed_dict={keep_prob:1.0,
                                                                          handle:test_iterator_handle})
#                                 pdb.set_trace()
                                test_predicted_ages.append(test_predicted_age)
                                test_labels.append(test_label)
                        except tf.errors.OutOfRangeError:
#                             pdb.set_trace()
                            test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                            test_labels = np.concatenate(tuple(test_labels))
                            test_loss = (get_loss(test_predicted_ages,test_labels)).eval()
                            test_losses.append(test_loss)
                            test_steps.append(step)
                            print('Step %d: training_loss = %.2f (%.3f sec)\n (val_loss)test_loss = %.2f' 
                                  %(step,loss_value,duration,test_loss))
                            
                            if test_loss < min_test_loss:
                                min_test_loss = test_loss
                                print('best shot model: test_loss = %.2f, step = %d' %(min_test_loss,step))
                                if step > 100:
                                    save_path = saver.save(sess, FLAGS.saver_dir)
                                    print('model saved successfully.')
                                    
                    steps.append(step)
                    losses.append(loss_value)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' %(num_epochs,step))
            
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
            ax1.plot(steps, losses)
            ax1.plot(test_steps,test_losses)
            ax1.set_title('trainning loss')
            ax1.grid(True)
            plt.savefig('./img/training_proc_zach.pdf', bbox_inches='tight')
            plt.savefig('./img/training_proc_zach.png', bbox_inches='tight')
            
            plt_data_path_name = './img/training_proc_zach_pltdata_' + time_now()
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
        
        prediction = convolutional_neural_network(X,keep_prob)
        loss = get_loss(prediction,label_batch)
    
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
                    test_predicted_age,test_label = sess.run([prediction, label_batch],
                                                    feed_dict={keep_prob:1.0,
                                                              handle:test_iterator_handle})
                    test_predicted_ages.append(test_predicted_age)
                    test_labels.append(test_label)
            except tf.errors.OutOfRangeError:
                test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                test_labels = np.concatenate(tuple(test_labels))
                
                test_loss = (get_loss(test_predicted_ages,test_labels)).eval()
                
                print('test_loss(RMSE) = %.2f.' 
                                  %(test_loss))
            fig = plt.figure()
            plt.title('Test Data')
            plt.xlabel('Chronological Age')
            plt.ylabel('Brain Age (Predicted)')
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            for i in range(len(test_labels)):
                plt.scatter(test_labels[i], test_predicted_ages[i], c = 'blue',s=1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig('./img/test_corr_zach.pdf', bbox_inches='tight')
            plt.savefig('./img/test_corr_zach.png', bbox_inches='tight')
            plt.show()
    return


def training_draw():
    with tf.Graph().as_default():
        handle = tf.placeholder(tf.string,shape=[])
        test_iterator = training_figure_iterator()[0]
        iterator = tf.data.Iterator.from_string_handle(handle, test_iterator.output_types)
        arr_batch,label_batch,id_batch = iterator.get_next()
        X = tf.reshape(arr_batch, [-1]+list(FLAGS.arr_shape)+[1])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        prediction = convolutional_neural_network(X,keep_prob)
        loss = get_loss(prediction,label_batch)
    
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
                    test_predicted_age,test_label = sess.run([prediction, label_batch],
                                                    feed_dict={keep_prob:1.0,
                                                              handle:test_iterator_handle})
                    test_predicted_ages.append(test_predicted_age)
                    test_labels.append(test_label)
            except tf.errors.OutOfRangeError:
                test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                test_labels = np.concatenate(tuple(test_labels))
                
                test_loss = (get_loss(test_predicted_ages,test_labels)).eval()
                
                print('test_loss(RMSE) = %.2f.' 
                                  %(test_loss))
            fig = plt.figure()
            plt.title('Test Data')
            plt.xlabel('Chronological Age')
            plt.ylabel('Brain Age (Predicted)')
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            for i in range(len(test_labels)):
                plt.scatter(test_labels[i], test_predicted_ages[i], c = 'blue',s=1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig('./img/training_corr_zach.pdf', bbox_inches='tight')
            plt.savefig('./img/training_corr_zach.png', bbox_inches='tight')
            plt.show()
    return



def main(_):
#     pdb.set_trace()
    arr = np.load('./data_npy/mean_npy.npy')
    FLAGS.arr_shape = arr.shape
    
    if not FLAGS.for_test:
        run_training()
    run_test()
    training_draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saver_dir',
                       type=str,
                       default="./log/model_zach.ckpt",
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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)