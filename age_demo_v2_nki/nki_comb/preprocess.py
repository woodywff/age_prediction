'''
Dataset: NKI
10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)


'''

import sys
sys.path.append("../..")
from dev_tools.my_tools import *
from dev_tools.preprocess_tools import *



def gen_tfrecord_comb(csv_path_name,npy_dir_raw,npy_dir_alff,tf_filename='tf.tfrecords'):
    '''
    To generate .tfrecord files according to .csv file
    
    csv_path_name: .csv file's path and name
    npy_dir: where are the .npy files
    tf_filename: name of .tfrecords file under ./
    gen_tfrecord_comb(csv_path_name,'../nki_raw/data_npy/mean_subtracted','../nki_alff/data_npy/origin',tf_filename='tf.tfrecords')
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
            npy_raw_path_filename = os.path.join(npy_dir_raw,npy_filename)
            npy_alff_path_filename = os.path.join(npy_dir_alff,npy_filename)
            
            try:
                arr_npy_raw = np.load(npy_raw_path_filename)
                arr_npy_alff = np.load(npy_alff_path_filename)
            except FileNotFoundError:
                print('No such file: ',npy_filename)
                continue
                
#             pdb.set_trace()
            label = round(age_list[i],2)
            arr_npy_raw = arr_npy_raw.astype(np.float32).tostring()
            arr_npy_alff = arr_npy_alff.astype(np.float32).tostring()
            
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'npy_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_npy_raw])),
                        'npy_alff': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_npy_alff])),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(sub_id)])),
                    }
                )
            )
            writer.write(example.SerializeToString())
            pbar.update(int(i*100/(n-1)))
        pbar.finish()
    return

def preprocess_main():
    '''
    '''
    print_sep('.tfrec generating starts')
    # get training.csv test.csv
    gen_training_test_csv()
    # get .tfrecords ready
    gen_tfrecord_comb('./training.csv','../nki_raw/data_npy/mean_subtracted','../nki_alff/data_npy/origin','training_data.tfrec')
    gen_tfrecord_comb('./test.csv','../nki_raw/data_npy/mean_subtracted','../nki_alff/data_npy/origin','test_data.tfrec')
    my_mkdir('./img')
    my_mkdir('./log')
    print_sep('.tfrec ready')
    return
    


if __name__ == '__main__':
    preprocess_main()
    
    