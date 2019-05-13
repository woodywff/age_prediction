import sys
sys.path.append("..")
from dev_tools.my_tools import *
from dev_tools.preprocess_tools import *

FLAGS_batch_size = 10

FLAGS_arr_shape_raw = np.load('../nki_raw/data_npy/mean_npy.npy').shape

FLAGS_arr_shape_alff = np.load('../nki_alff/data_npy/mean_npy.npy').shape


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
    npy_raw = tf.reshape(npy_raw,list(FLAGS_arr_shape_raw))
    npy_alff = tf.decode_raw(features['npy_alff'],tf.float32)
    npy_alff = tf.reshape(npy_alff,list(FLAGS_arr_shape_alff))
    label = features['label']
    sub_id = features['id']
    
    return npy_raw, npy_alff,label,sub_id




def get_iterator(for_training=True,num_epochs=1):
    if not num_epochs:
        num_epochs = None
    filename = os.path.join('./',('training_data' if for_training else 'test_data')+'.tfrec')
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
#         pdb.set_trace()
        dataset = dataset.map(decode)
        if for_training:
            dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(FLAGS_batch_size)
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
    return_list: [raw_batch,alff_batch,label_batch,id_batch, X]
    
    mean_version: to use mean_subtracted or origin .npy
    '''
    decode_arr_raw = return_list[0]
    decode_arr_alff = return_list[1]
    decode_label = return_list[2]
    decode_label = np.round(decode_label,2)
    decode_id = return_list[3]
    decode_X_raw = return_list[4]
    decode_X_alff = return_list[5]
    return_value = True

    for j in range(decode_id.shape[0]):
        d_a_raw = decode_arr_raw[j]
        d_a_alff = decode_arr_alff[j]
        d_l = decode_label[j]
        d_i = decode_id[j]
        
        d_x_raw = decode_X_raw[j]
        d_x_raw = np.reshape(d_x_raw,FLAGS_arr_shape_raw)
        
        d_x_alff = decode_X_alff[j]
        d_x_alff = np.reshape(d_x_alff,FLAGS_arr_shape_alff)


        npy_filename = str(d_i) + '.npy'
        npy_path_filename_raw = os.path.join('../nki_raw/data_npy/mean_subtracted/',npy_filename)
        npy_path_filename_alff = os.path.join('../nki_alff/data_npy/origin/',npy_filename)

        try:
            arr_npy_raw = np.load(npy_path_filename_raw)
            arr_npy_alff = np.load(npy_path_filename_alff)
        except FileNotFoundError:
            print('No such file: ',npy_filename)
            continue
        label = round(info_df['age'][info_df['id']==d_i].values[0],2)
        arr_npy_raw = arr_npy_raw.astype(np.float32)
        arr_npy_alff = arr_npy_alff.astype(np.float32)

        if np.sum(np.round(d_x_raw-arr_npy_raw,2)) != 0:
            print('d_x_raw != arr_npy_raw')
            return_value = False
        if np.sum(np.round(d_x_alff-arr_npy_alff,2)) != 0:
            print('d_x_alff != arr_npy_alff')
            return_value = False


        if round(label-d_l,1)!=0 or np.sum(d_a_raw!=arr_npy_raw)!=0 or np.sum(d_a_alff!=arr_npy_alff)!=0:
            print(npy_path_filename)
            print('original label: ',label)
            print('extracted label: ',d_l)
            print('label == decode_label: ',round(label-d_l,2)==0)
            print('arr_raw == decode_arr_raw: ',np.sum(d_a_raw!=arr_npy_raw))
            print('arr_alff == decode_arr_alff: ',np.sum(d_a_alff!=arr_npy_alff))
            return_value = False
    #         break


#         if d_i == 51075:
#             print_sep()
#     print('try_equal finished.')
    return return_value




def tf_varify(num_epochs=3):
    '''
    '''
    
    iterators = get_iterator(num_epochs=num_epochs)
    test_iterator = get_iterator(for_training=False)[0]
    
    handle = tf.placeholder(tf.string,shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, iterators[0].output_types)
    raw_batch,alff_batch,label_batch,id_batch = iterator.get_next()


    X_raw = tf.reshape(raw_batch, [-1]+list(FLAGS_arr_shape_raw)+[1])
    X_alff = tf.reshape(alff_batch, [-1]+list(FLAGS_arr_shape_alff)+[1])

    info_df_training = pd.read_csv('./training.csv', sep=',',header=0)
    info_df_test = pd.read_csv('./test.csv', sep=',',header=0)

    with tf.Session() as sess:
        train_iterator_handle = sess.run(iterators[0].string_handle())
        val_iterator_handle = sess.run(iterators[1].string_handle())
        test_iterator_handle = sess.run(test_iterator.string_handle())
        
        try:
            step = 0
            while True:
                start_time = time.time()
                return_list_training = sess.run([raw_batch,alff_batch,label_batch,id_batch,X_raw,X_alff],
                                                feed_dict={handle:train_iterator_handle})
                if not try_equal(return_list_training,info_df_training):
                    print('ERROR: Where am I: in return_list_training.')

                return_list_val = sess.run([raw_batch,alff_batch,label_batch,id_batch,X_raw,X_alff], 
                                           feed_dict={handle:val_iterator_handle})
                if not try_equal(return_list_val,info_df_training):
                    print('ERROR: Where am I: in return_list_val.')

                if step % 100 == 0:
                    sess.run(test_iterator.initializer)
                    try:
                        while True:
                            return_list_test = sess.run([raw_batch,alff_batch,label_batch,id_batch,X_raw,X_alff], 
                                                       feed_dict={handle:test_iterator_handle})
                            if not try_equal(return_list_test,info_df_test):
                                print('ERROR: Where am I: in return_list_test.')
                    except tf.errors.OutOfRangeError:
                        print('test finished inside training')
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' %(num_epochs,step))
            
    return


if __name__ == '__main__':
    tf_varify()