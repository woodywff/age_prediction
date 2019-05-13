# to restore two networks
import sys
sys.path.append("../")
sys.path.append("../..")
from dev_tools.model_tools import *
from dev_tools.preprocess_tools import *
import argparse

FLAGS = None

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def inference4comb(X, keep_prob=1.0, is_training_forBN=False, trivial=True, FLAGS_arr_shape=(67,67,67)):
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
        
#         print(fc_out.name, id(fc_out))
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
    
    return final_output, l2_loss, fc_out
        
def get_loss(predict_batches,label_batches):
    '''
    we are not sure the shape of predict_batches,label_batches, (?,1) or (?,),
    so we reshape them into (?,) first.
    '''
    with tf.name_scope('cross_entropy'):
        predict_batches = tf.reshape(predict_batches,[-1,1])
        label_batches = tf.reshape(label_batches,[-1,1])
        cost = tf.reduce_mean(tf.square(predict_batches - label_batches))
    return cost



def decode(serialized_example):
    '''
    serialized_example: tf.data.TFRecordDataset
    '''
    features = tf.parse_single_example(
        serialized_example,
        features={
            'npy_raw':tf.FixedLenFeature([],tf.string),
            'npy_alff':tf.FixedLenFeature([],tf.string),
            'label': tf.FixedLenFeature([],tf.float32),
            'id': tf.FixedLenFeature([],tf.int64)
        }
    )
    npy_raw = tf.decode_raw(features['npy_raw'],tf.float32)
    npy_raw = tf.reshape(npy_raw,list(FLAGS.arr_shape_raw))
    npy_alff = tf.decode_raw(features['npy_alff'],tf.float32)
    npy_alff = tf.reshape(npy_alff,list(FLAGS.arr_shape_alff))
    label = features['label']
    sub_id = features['id']
    
    return npy_raw, npy_alff,label,sub_id


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


def restored_variable_check(sess,model_1_ckpt,model_2_ckpt):
    '''
    '''
    # Variable check
    param_list = tf.global_variables()
    reader_raw = pywrap_tensorflow.NewCheckpointReader(model_1_ckpt)
    reader_alff = pywrap_tensorflow.NewCheckpointReader(model_2_ckpt)

    pbar = ProgressBar().start()
    n_bar = len(param_list)
    for i,param in enumerate(param_list):
        if 'alff/' in param.name:
            if np.sum(sess.graph.get_tensor_by_name(param.name).eval() - 
                  reader_alff.get_tensor(param.name.split(':')[0])) != 0:
                print('\033[31m',param)
        elif 'raw/' in param.name:
            if np.sum(sess.graph.get_tensor_by_name(param.name).eval() - 
                  reader_raw.get_tensor(param.name.split(':')[0])) != 0:
                print('\033[31m',param)
        else:
            print('\033[31mUnknown variable: ',param)
    #                 break
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    print('Check finished.')
    return


def run_training():
    return_list = []
    MSE_SAVED = False
    PERSON_SAVED = False
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            num_epochs = FLAGS.num_epochs
            iterators = get_iterator(num_epochs=num_epochs)
            handle = tf.placeholder(tf.string,shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, iterators[0].output_types)
            test_iterator = get_iterator(for_training=False)[0]
            raw_batch,alff_batch,label_batch,_ = iterator.get_next()
            
            with tf.variable_scope('raw'):
                X_raw = tf.reshape(raw_batch, [-1]+list(FLAGS.arr_shape_raw)+[1])
                _,_,fc_out_raw = inference4comb(X_raw,trivial=False,
                                               FLAGS_arr_shape=FLAGS.arr_shape_raw)

                saver = tf.train.Saver([p_name for p_name in tf.global_variables() if 'raw/' in p_name.name])
                saver.restore(sess, FLAGS.model_1)

            with tf.variable_scope('alff'):
                X_alff = tf.reshape(alff_batch, [-1]+list(FLAGS.arr_shape_alff)+[1])
                _,_,fc_out_alff = inference4comb(X_alff,trivial=False,
                                                  FLAGS_arr_shape=FLAGS.arr_shape_alff)

                saver = tf.train.Saver([p_name for p_name in tf.global_variables() if 'alff/' in p_name.name])
                saver.restore(sess, FLAGS.model_2)

            # Variable check
            if FLAGS.variable_check:
                restored_variable_check(sess,FLAGS.model_1,FLAGS.model_2)

            with tf.name_scope('last_layer'):
                w = tf.Variable(tf.truncated_normal([256,1],stddev=0.1),name='w')
                b = tf.Variable(tf.constant(0.1,shape=[1]),name='b')
                predicted_age = tf.add(tf.matmul(tf.concat([fc_out_raw,fc_out_alff],1),w), b, name='final_output')
            
            loss = get_loss(predicted_age,label_batch)
            optimizer = tf.train.AdamOptimizer(1e-4)

            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     "last_layer")
#             print(train_vars)
            train_op = optimizer.minimize(loss, var_list=train_vars)

            # Variable init
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
#             print(uninitialized_vars)
            initialize_op = tf.variables_initializer(uninitialized_vars)
            sess.run(initialize_op)
            
#             print([x for x in tf.get_default_graph().get_operations() if 'Placeholder' in x.type])
            
            saver = tf.train.Saver()
            losses = []
            acces = []
            steps = []
            
            test_losses = []
            test_acces = []
            test_steps = []
            
            train_iterator_handle = sess.run(iterators[0].string_handle())
            val_iterator_handle = sess.run(iterators[1].string_handle())
            test_iterator_handle = sess.run(test_iterator.string_handle())
            
            print('Let\'s get started to train!')
            start_time = time.time()
            try:
                step = 0
                min_val_mse = FLAGS.MIN_VAL_MSE
                max_val_person = FLAGS.MAX_VAL_PERSON
                while True:
                    sess.run(train_op, {handle:train_iterator_handle})

                    loss_value,val_pred_age,val_chro_age = sess.run([loss,predicted_age,label_batch],
                                                                    {handle:val_iterator_handle})
                    acc_value = person_corr(val_pred_age,val_chro_age)
                    if step % 100 == 0:
                        sess.run(test_iterator.initializer)
                        test_predicted_ages = []
                        test_labels = []
                        try:
                            while True:
                                test_predicted_age,test_label = sess.run([predicted_age,label_batch],
                                                                         {handle:test_iterator_handle})
                                test_predicted_ages.append(test_predicted_age)
                                test_labels.append(test_label)
                        except tf.errors.OutOfRangeError:
#                             pdb.set_trace()
                            test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
                            test_labels = np.concatenate(tuple(test_labels))
                            test_loss = (get_loss(test_predicted_ages,test_labels)).eval()
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
            
    return 


def test_sess(input_iterator,model_path):
    '''
    main part of test process
    model_path: str; directry of the saved model which to be loaded, the best mse or the best person.
    
    '''
    handle = tf.placeholder(tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, input_iterator.output_types)
    raw_batch,alff_batch,label_batch,_ = iterator.get_next()

    with tf.variable_scope('raw'):
        X_raw = tf.reshape(raw_batch, [-1]+list(FLAGS.arr_shape_raw)+[1])
        _,_,fc_out_raw = inference4comb(X_raw,trivial=False,
                                       FLAGS_arr_shape=FLAGS.arr_shape_raw)

    with tf.variable_scope('alff'):
        X_alff = tf.reshape(alff_batch, [-1]+list(FLAGS.arr_shape_alff)+[1])
        _,_,fc_out_alff = inference4comb(X_alff,trivial=False,
                                          FLAGS_arr_shape=FLAGS.arr_shape_alff)

    with tf.name_scope('last_layer'):
        w = tf.Variable(tf.truncated_normal([256,1],stddev=0.1),name='w')
        b = tf.Variable(tf.constant(0.1,shape=[1]),name='b')
        predicted_age = tf.add(tf.matmul(tf.concat([fc_out_raw,fc_out_alff],1),w), b, name='final_output')

    loss = get_loss(predicted_age,label_batch)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        print('Model loaded successfully.')
        
#         pdb.set_trace()
        if FLAGS.variable_check:
            restored_variable_check(sess,FLAGS.model_1,FLAGS.model_2)
        
        
        test_iterator_handle = sess.run(input_iterator.string_handle())
        sess.run(input_iterator.initializer)
        test_predicted_ages = []
        test_labels = []
        try:
            while True:
                test_predicted_age,test_label = sess.run([predicted_age, label_batch],
                                                         {handle:test_iterator_handle})
                test_predicted_ages.append(test_predicted_age)
                test_labels.append(test_label)
        except tf.errors.OutOfRangeError:
            test_predicted_ages = np.concatenate(tuple(test_predicted_ages))
            test_labels = np.concatenate(tuple(test_labels))

            test_loss = (get_loss(test_predicted_ages,test_labels)).eval()
            test_acc = person_corr(test_predicted_ages,test_labels)
            test_mae = calc_mae(test_predicted_ages,test_labels)

            print('MSE = %.2f, MAE = %.2f, Person correlation coefficient = %.2f.' 
                              %(test_loss,test_mae,test_acc))
    return test_loss,test_acc,test_mae,test_predicted_ages,test_labels


def test_training_set(model_path):
    '''
    test process, training set as input
    model_path: str; directry of the saved model which to be loaded, the best mse or the best person.
    '''
    with tf.Graph().as_default():
        iterator = training_figure_iterator()[0]
#         test_sess(iterator,model_path)
        test_loss, test_acc, test_mae, pred_age, chro_age= test_sess(iterator,model_path)
        draw_person_corr(pred_age,chro_age,test_loss,test_acc,test_mae,title='Training Data',
                         save_filename='training_corr_'+model_path.split('.')[1].split('_')[-1]+'_woody.pdf')
    return

def test_test_set(model_path):
    '''
    test process, test set as input
    model_path: str; directry of the saved model which to be loaded, the best mse or the best person.
    '''
    with tf.Graph().as_default():
        iterator = get_iterator(for_training=False)[0]
        test_loss, test_acc, test_mae, pred_age, chro_age= test_sess(iterator,model_path)
        draw_person_corr(pred_age,chro_age,test_loss,test_acc,test_mae,title='Test Data',
                         save_filename='test_corr_'+model_path.split('.')[1].split('_')[-1]+'_woody.pdf')
    return




def main(_):
#     pdb.set_trace()
    FLAGS.arr_shape_raw = np.load('../nki_raw/data_npy/mean_npy.npy').shape

    FLAGS.arr_shape_alff = np.load('../nki_alff/data_npy/mean_npy.npy').shape
    
    if not FLAGS.for_test:
        run_training()
    test_training_set(FLAGS.saver_dir_mse)
    test_training_set(FLAGS.saver_dir_person)
    test_test_set(FLAGS.saver_dir_mse)
    test_test_set(FLAGS.saver_dir_person)

# python model_woody_comb.py --model_1=<.ckpt> --model_2=<.ckpt> 
#                            --saver_dir_mse=<best mse.ckpt> --saver_dir_person=<best pearson.ckpt>
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1',
                       type=str,
                       default="./log/model_mse_woody_raw.ckpt",
                       help='model 1\' checkpoint.')
    parser.add_argument('--model_2',
                       type=str,
                       default="./log/model_mse_woody_alff.ckpt",
                       help='model 2\' checkpoint.')
    parser.add_argument('--saver_dir_mse',
                       type=str,
                       default="./log/comb_mse_raw_alff_mse.ckpt",
                       help='Directory to save checkpoint.')
    parser.add_argument('--saver_dir_person',
                       type=str,
                       default="./log/comb_mse_raw_alff_pearson.ckpt",
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
                        default=200,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--MIN_VAL_MSE',
                       type=float,
                       default=300.0,
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
    parser.add_argument('--variable_check',
                       type=bool,
                       default=False,
                       help='Have a check on the restored variables.')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)