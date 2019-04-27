'''


'''
import sys
sys.path.append("..")
from dev_tools.my_tools import *
from dev_tools.preprocess_tools import *

# DESIRED_SHAPE=(130, 130, 110)
# DESIRED_SHAPE=(130, 130, 130)
DESIRED_SHAPE = (67,67,67)

#-------------------- once for a life thing (dataset specific)-------------------------------

def extract_tgz():
    '''
    *** (dataset specific) ***
    extract .tgz files
    '''
    source_path = '/media/woody/Elements/age_data/ABIDE_II'
    target_path = os.path.join(source_path,'data')
    my_mkdir(target_path)
    file_list = os.listdir(source_path)
    pbar = ProgressBar().start()
    n_bar = len(file_list)
    
    for i,filename in enumerate(file_list):
        re_result = re.match('^.*\.tar\.gz$',filename)
        if re_result:
            tgz_path_name = os.path.join(source_path,filename)
            with tarfile.open(tgz_path_name) as tar:
                names = tar.getnames()
                for name in names:
                    if not os.path.exists(os.path.join(target_path,name)):
                        tar.extract(name, target_path)
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    return

def orgnize_files():
    '''
    *** (dataset specific) ***
    copy .nii.gz from different centers folders to the target_path and rename them as <sub_id>.nii.gz
    '''
    source_path = '/media/woody/Elements/age_data/ABIDE_II/data/'
    target_path = '/media/woody/Elements/age_data/ABIDE_II/anat_data/'
    my_mkdir(target_path)
    centers = os.listdir(source_path)
    
    for center in centers:
        subid_path = os.path.join(source_path,center)
        sub_ids = os.listdir(subid_path)
        pbar = ProgressBar().start()
        n_bar = len(sub_ids)
        for i,sub_id in enumerate(sub_ids):
            src_path_name = os.path.join(subid_path,sub_id,'session_1','anat_1','anat.nii.gz')
            dst_path_name = os.path.join(target_path,str(int(sub_id)) + '.nii.gz')
            if not os.path.exists(dst_path_name):
                try:
                    shutil.copyfile(src_path_name,dst_path_name)
                except FileNotFoundError:
                    try:
                        shutil.copyfile(os.path.join(subid_path,sub_id,'session_1','anat_1','anat_rpi.nii.gz'),dst_path_name)
                    except FileNotFoundError:
                        try:
                            shutil.copyfile(os.path.join(subid_path,sub_id,'session_2','anat_1','anat.nii.gz'),dst_path_name)
                        except FileNotFoundError:
                            print(src_path_name, 'not found.')
            pbar.update(int(i*100/(n_bar-1)))
        pbar.finish()
    return

def gen_phenotypics():
    '''
    *** (dataset specific) ***
    This is an once for all thing.
    To generate phenotypics.csv with id and age only
    As the assumption of the model is that healthy individuals' chronological ages are close to their brain ages,
    all brain images from individuals with autism and Alzheimer's disease were excluded.
    '''
    if os.path.exists('./phenotypics.csv'):
        print('phenotypics.csv exists already.')
        return
    original_csv_path_name = '/media/woody/Elements/age_data/ABIDE_II/ABIDEII_Composite_Phenotypic_utf8.csv'
    phenotypics = pd.read_csv(original_csv_path_name, sep=',',header=0)

    autism_csv = phenotypics[phenotypics['DX_GROUP'].isin([1])]
    control_csv = phenotypics[phenotypics['DX_GROUP'].isin([2])]

    id_list = list(control_csv['SUB_ID'])
    age_list = list(control_csv['AGE_AT_SCAN'])

    # shuffle and save the phenotypic info:
    id_list, age_list = get_shuffled(id_list,age_list)
    data_to_save = pd.DataFrame({'id':id_list,'age':age_list})
    data_to_save.to_csv(os.path.join('./','phenotypics.csv'), index=False, sep=',')
    print('phenotypics.csv created.')
    return

#-------------------------------- END once for a life thing (dataset specific) -------------------------------------------------------

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

#------------------------------------End of crop and pad for ABIDE-----------------------------


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
        re_result = re.match('^.*\.nii\.gz$',filename)
        if re_result:
            target_filename = os.path.join(target_dir_origin,filename.split('.')[0])
            if not os.path.exists(target_filename + '.npy'):
                cropped_npy = inner_preprocess_1(os.path.join(source_dir,filename))
                cropped_npy = minmax_normalize(cropped_npy)
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
    resampled_img = resample(npy_img, pixdim, [4,4,4])    # [4,4,4] decides the resolution is shrinked to 1/4 of origin
#     print('resampled img shape: ',resampled_img.shape)
    crop_padded_img = crop_pad_abide(resampled_img,DESIRED_SHAPE) # this one is for cropping scenario 
#     crop_padded_img = crop_pad(resampled_img,DESIRED_SHAPE) # this one is only for padding without cropping operation
                                                            # this one is more general, so if this works we don't need to use 
                                                            # the above one.
#     print('crop and padded img shape: ', crop_padded_img.shape)
    crop_padded_img = np.round(crop_padded_img)
    return crop_padded_img.astype(int)

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
    std_npy = calc_std()
    
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
#                 subtracted_npy = origin_npy - mean_npy
                subtracted_npy = (origin_npy - mean_npy)/std_npy
                np.save(os.path.join(target_dir_mean,filename.split('.')[0]),subtracted_npy)
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    
    return

    
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
        my_mkdir(path)
    
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
            npy_path_filename = os.path.join(npy_dir,npy_filename)
            try:
                arr_npy = np.load(npy_path_filename)
            except FileNotFoundError:
                print('No such file: ',npy_filename)
                continue
                
#             pdb.set_trace()
            label = round(age_list[i],2)
            arr_raw = arr_npy.tostring()
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'arr_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw])),
                        'label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(sub_id)])),
                    }
                )
            )
            writer.write(example.SerializeToString())
            pbar.update(int(i*100/(n-1)))
        pbar.finish()
    return

def preprocess_run():
    # extract_tgz() # this function should be commented after the first run
    # orgnize_files() # this function should be commented after the first run
    
    # things above happens on the mobile HDD
    # things below happens under ./
    print_sep('preprocessing starts')
    # get phenotypics.csv
    gen_phenotypics()
    # get training.csv test.csv
    gen_training_test_csv()
    # get preprocessed .npy files
    # please refer to the comments of gen_npy()
    source_dir = '/media/woody/Elements/age_data/ABIDE_II/anat_data/'
    target_dir = './data_npy'
    gen_npy(source_dir,target_dir)
    # # get .tfrecords ready
    gen_tfrecord('./training.csv',npy_dir='./data_npy/mean_subtracted/',tf_filename='training_data.tfrec')
    gen_tfrecord('./test.csv',npy_dir='./data_npy/mean_subtracted/',tf_filename='test_data.tfrec')

    my_mkdir('./img')
    my_mkdir('./log')
    print_sep('preprocessing FINISHED')
    return
    

# this is the entrance of this file
if __name__ == '__main__':
    preprocess_run()
    
    