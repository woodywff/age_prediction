'''
10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)

Using 3D-CNN model code of my own
'''
from preprocess import *
from model_tools import *
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
    '''
    we are not sure the shape of predict_batches,label_batches, (?,1) or (?,),
    so we reshape them into (?,) first.
    '''
    with tf.name_scope('cross_entropy'):
        predict_batches = tf.reshape(predict_batches,[-1,1])
        label_batches = tf.reshape(label_batches,[-1,1])
        cost = tf.reduce_mean(tf.square(predict_batches - label_batches)) + FLAGS.l2_epsilon * l2_loss
    return cost


def person_corr(predicts,labels):
    '''
    (get_acc())
    we are not sure the shape of predicts,labels, (?,1) or (?,),
    so we reshape them into (?,) first.
    '''
    return np.corrcoef(np.vstack((predicts.reshape(-1),labels.reshape(-1))))[0,1]



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
    '''
    serialized_example: tf.data.TFRecordDataset
    '''
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
    
    
    
    
def run_training():
    '''
    training set: training_data.tfrec
    valadation set: test_data.tfrec
    
    there are three iterator handles:
        train_iterator_handle: for training
        val_iterator_handle: for calculation and drawing of the training set's mse and person correlation coefficient
        test_iterator_handle: for validation
    '''
    return_list = []
    MSE_SAVED = False
    PERSON_SAVED = False
    
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
            start_time = time.time()
            
            try:
                step = 0
                min_val_mse = FLAGS.MIN_VAL_MSE
                max_val_person = FLAGS.MAX_VAL_PERSON
                while True:
                    sess.run(train_op, feed_dict={keep_prob:0.5,
                                                 is_training_forBN:True,
                                                handle:train_iterator_handle})

                    loss_value,val_pred_age,val_chro_age = sess.run([loss,predicted_age,label_batch],feed_dict={keep_prob:1.0,
                                                       is_training_forBN:False,
                                                      handle:val_iterator_handle})
                    acc_value = person_corr(val_pred_age,val_chro_age)
                    if step % 100 == 0:
                        sess.run(test_iterator.initializer)
                        test_predicted_ages = []
                        test_labels = []
                        try:
                            while True:
                                test_predicted_age,test_label,test_id = sess.run([predicted_age,label_batch,id_batch],
                                                                feed_dict={keep_prob:1.0,
                                                                           is_training_forBN:False,
                                                                          handle:test_iterator_handle})
#                                 pdb.set_trace()
#                                 print("\033[0;30;40m\tTRY:\033[0m")
#                                 print(test_id)
                                test_predicted_ages.append(test_predicted_age)
                                test_labels.append(test_label)
                        except tf.errors.OutOfRangeError:
#                             pdb.set_trace()
                            test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                            test_labels = np.concatenate(tuple(test_labels))
                            test_loss = (get_loss(test_predicted_ages,test_labels,tf.constant(0.0))).eval()
                            test_acc = person_corr(test_predicted_ages,test_labels)
                            test_losses.append(test_loss)
                            test_acces.append(test_acc)
                            test_steps.append(step)
                            print('Step %d: (loss func: MSE, acc func: Pearson correlation coefficient)\n \
                            training_loss = %.2f, training_acc = %.2f\n \
                            val_loss(test_loss) = %.2f, val_acc(test_acc) = %.2f' 
                                  %(step,loss_value,acc_value,test_loss,test_acc))
                            
                            if test_loss < min_val_mse:
                                min_val_mse = test_loss
                                print('best mse model: mse of validation set = %.2f, step = %d' %(min_val_mse,step))
                                if step > FLAGS.MIN_SAVED_STEP:
                                    saver.save(sess, FLAGS.saver_dir_mse)
                                    print('best mse model saved successfully.')
                                    MSE_SAVED = True
                            if test_acc > max_val_person:
                                max_val_person = test_acc
                                print('best person correlation model: person correlation coefficient of validation set = %.2f, step = %d' %(max_val_person,step))
                                if step > FLAGS.MIN_SAVED_STEP:
                                    save_path = saver.save(sess, FLAGS.saver_dir_person)
                                    print('best person correlation model saved successfully.')
                                    PERSON_SAVED = True
                                    
                    steps.append(step)
                    acces.append(acc_value)
                    losses.append(loss_value)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' %(num_epochs,step))
                end_time = time.time()
                print('time elapsed: '+sec2hms(end_time-start_time))
                return_list = [steps,losses,acces,test_steps,test_losses,test_acces]
                return_list_path_name = './img/training_return_list_woody'
                if os.path.exists(return_list_path_name + '.npy'):
                    print(return_list_path_name + '.npy exists already. Cover it...')
                np.save(return_list_path_name, np.array(return_list))
                
                if not MSE_SAVED:
                    saver.save(sess, FLAGS.saver_dir_mse)
                    print('Save the last model as the best mse model.')
                if not PERSON_SAVED:
                    saver.save(sess, FLAGS.saver_dir_person)
                    print('Save the last model as the best person correlation model.')

    if len(return_list) == 0:
        print('Something wrong with training process...')
        return -1
    else:
        return 0 


def test_sess(input_iterator,model_path):
    '''
    main part of test process
    model_path: str; directry of the saved model which to be loaded, the best mse or the best person.
    
    '''
    handle = tf.placeholder(tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, input_iterator.output_types)
    arr_batch,label_batch,id_batch = iterator.get_next()
    X = tf.reshape(arr_batch, [-1]+list(FLAGS.arr_shape)+[1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    is_training_forBN = tf.placeholder(tf.bool, name='is_training_forBN')

    predicted_age,l2_loss = inference(X,keep_prob,is_training_forBN)

    loss = get_loss(predicted_age,label_batch,l2_loss)

    saver = tf.train.Saver()
#         pdb.set_trace()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        print('Model loaded successfully.')
        test_iterator_handle = sess.run(input_iterator.string_handle())
        sess.run(input_iterator.initializer)
        test_predicted_ages = []
        test_labels = []
        try:
            while True:
                test_predicted_age,test_label,test_id = sess.run([predicted_age, label_batch,id_batch],
                                                feed_dict={keep_prob:1.0,
                                                           is_training_forBN:False,
                                                          handle:test_iterator_handle})
#                 pdb.set_trace()
#                 print("\033[0;30;40m\tTRY:\033[0m")
#                 print(test_id)
                test_predicted_ages.append(test_predicted_age)
                test_labels.append(test_label)
        except tf.errors.OutOfRangeError:
            test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
            test_labels = np.concatenate(tuple(test_labels))

            test_loss = (get_loss(test_predicted_ages,test_labels,tf.constant(0.0))).eval()
            test_acc = person_corr(test_predicted_ages,test_labels)

            print('test_loss = %.2f, test_accuracy = %.2f.' 
                              %(test_loss,test_acc))
    return test_loss,test_acc,test_predicted_ages,test_labels



def test_training_set(model_path):
    '''
    test process, training set as input
    model_path: str; directry of the saved model which to be loaded, the best mse or the best person.
    '''
    with tf.Graph().as_default():
        iterator = training_figure_iterator()[0]
        test_loss, test_acc, pred_age, chro_age= test_sess(iterator,model_path)
        draw_person_corr(pred_age,chro_age,test_loss,test_acc,title='Training Data',
                         save_filename='training_corr_'+model_path.split('_')[1]+'_woody.pdf')
    return

def test_test_set(model_path):
    '''
    test process, test set as input
    model_path: str; directry of the saved model which to be loaded, the best mse or the best person.
    '''
    with tf.Graph().as_default():
        iterator = get_iterator(for_training=False)[0]
        test_loss, test_acc, pred_age, chro_age= test_sess(iterator,model_path)
        draw_person_corr(pred_age,chro_age,test_loss,test_acc,title='Test Data',
                         save_filename='test_corr_'+model_path.split('_')[1]+'_woody.pdf')
    return




def main(_):
#     pdb.set_trace()
    arr = np.load('./data_npy/mean_npy.npy')
    FLAGS.arr_shape = arr.shape
    
    if not FLAGS.for_test:
        run_training()
    test_training_set(FLAGS.saver_dir_mse)
    test_training_set(FLAGS.saver_dir_person)
    test_test_set(FLAGS.saver_dir_mse)
    test_test_set(FLAGS.saver_dir_person)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saver_dir_mse',
                       type=str,
                       default="./log/model_mse_woody.ckpt",
                       help='Directory to save checkpoint.')
    parser.add_argument('--saver_dir_person',
                       type=str,
                       default="./log/model_person_woody.ckpt",
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
    parser.add_argument('--MIN_VAL_MSE',
                       type=float,
                       default=100.0,
                       help='For validation set, the minimum loss value less than which the model would be saved.')
    parser.add_argument('--MAX_VAL_PERSON',
                       type=float,
                       default=0.6,
                       help='For validation set, the maximum accuracy value (person correlation)\
                        greater than which the model would be saved.')
    parser.add_argument('--MIN_SAVED_STEP',
                       type=int,
                       default=2000,
                       help='After which step the model is possible to be saved.')
    parser.add_argument('--for_test',
                       type=bool,
                       default=False,
                       help='If it is True, the program only runs the test part.')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)