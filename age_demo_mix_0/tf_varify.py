'''
This .py provides the process of check the compliance between the read data from .tfrec and the source data.
'''
from preprocess import *

NUM_EPOCHS = 10
SHAPE = np.load('./data_npy/mean_npy.npy').shape
BATCH_SIZE = 10



def decode(serialized_example):
    '''
    to decode data from .tfrec files
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
    arr = tf.reshape(arr,list(SHAPE))
    arr = tf.cast(arr,tf.float32) # to be compliance with the restriction of TypeError: 
                                  # Value passed to parameter 'input' has DataType int64 not in list of allowed values: 
                                  # float16, bfloat16, float32, float64
    label = features['label']
    sub_id = features['id']
    
    return arr,label,sub_id



def get_iterator(for_training=True,num_epochs=1):
    '''
    to generate iterators for training, validation and test datasets
    '''
    if not num_epochs:
        num_epochs = None
    root_dir = './'
    filename = os.path.join(root_dir,('training_data' if for_training else 'test_data')+'.tfrec')

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
#         pdb.set_trace()
        dataset = dataset.map(decode)
        if for_training:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(BATCH_SIZE)
        if for_training:
            train_iterator = dataset.make_one_shot_iterator()
            val_iterator = dataset.make_one_shot_iterator()
            iterators = [train_iterator,val_iterator]
        else:
            test_iterator = dataset.make_initializable_iterator()
            iterators = [test_iterator]

        
    return iterators

def try_equal(return_list,info_df):
    '''
    this function is going to print the data's information out when there's difference compared with the source data.
    return_list: [arr_batch,label_batch,id_batch, X]
    '''
    npy_dir='./data_npy/mean_subtracted/'
    decode_arr = return_list[0]
    decode_arr = decode_arr.astype(np.int64)

    decode_label = return_list[1]
    decode_label = np.round(decode_label,2)
    decode_id = return_list[2]
    decode_X = return_list[3]
    return_value = True

    for j in range(decode_id.shape[0]):
        d_a = decode_arr[j]
        d_l = decode_label[j]
        d_i = decode_id[j]

        d_x = decode_X[j]
        d_x = np.reshape(d_x,SHAPE)

        npy_filename = str(d_i) + '.npy'
        npy_path_filename = os.path.join(npy_dir,npy_filename)
#         print(npy_filename)

        try:
            arr_npy = np.load(npy_path_filename)
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        label = round(info_df['age'][info_df['id']==d_i].values[0],2)
#         arr = arr_npy.astype(np.float32)
        arr=arr_npy

        if np.sum(d_x!=arr) != 0:
            print('d_x != arr')
            return_value = False


        if round(label-d_l,1)!=0 or np.sum(d_a!=arr)!=0:
            print(npy_path_filename)
            print('original label: ',label)
            print('extracted label: ',d_l)
            print('label == decode_label: ',round(label-d_l,2)==0)
            print('arr == decode_arr: ',np.sum(d_a!=arr))
            return_value = False
    #         break


        if d_i == 51075:
            print_sep()
#     print('try_equal finished.')
    return return_value




def tf_varify():
    '''
    This session checks the difference between data read from .tfrec files and their corresponding source data.
    for training, validation and test sets respectively
    '''
    iterators = get_iterator(num_epochs=NUM_EPOCHS)
    handle = tf.placeholder(tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, iterators[0].output_types)
    arr_batch,label_batch,id_batch = iterator.get_next()


    X = tf.reshape(arr_batch, [-1]+list(SHAPE)+[1])

    info_df_training = pd.read_csv('./training.csv', sep=',',header=0)
    info_df_test = pd.read_csv('./test.csv', sep=',',header=0)

    with tf.Session() as sess:
        train_iterator_handle = sess.run(iterators[0].string_handle())
        val_iterator_handle = sess.run(iterators[1].string_handle())

        test_iterator = get_iterator(for_training=False)[0]
        test_iterator_handle = sess.run(test_iterator.string_handle())


        try:
            step = 0
            while True:
                start_time = time.time()
                return_list_training = sess.run([arr_batch,label_batch,id_batch, X], feed_dict={handle:train_iterator_handle})
                
#                 pdb.set_trace()
                if not try_equal(return_list_training,info_df_training):
                    print('Where am I: in return_list_training.')
                return_list_val = sess.run([arr_batch,label_batch,id_batch, X], feed_dict={handle:val_iterator_handle})
#                 pdb.set_trace()
                if not try_equal(return_list_val,info_df_training):
                    print('Where am I: in return_list_val.')
                duration = time.time() - start_time
#                     print(type(loss_value))
                if step % 100 == 0:
                    sess.run(test_iterator.initializer)
                    try:
                        while True:
                            return_list_test = sess.run([arr_batch,label_batch,id_batch, X], 
                                                       feed_dict={handle:test_iterator_handle})
                            
#                             pdb.set_trace()
                            if not try_equal(return_list_test,info_df_test):
                                print('Where am I: in return_list_test.')
                    except tf.errors.OutOfRangeError:
                        print('test finished inside training')
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' %(NUM_EPOCHS,step))
            
    return


if __name__ == '__main__':
    tf_varify()