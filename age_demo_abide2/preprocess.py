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
                cropped_npy = inner_preprocess_1(os.path.join(source_dir,filename),abide=True)
                cropped_npy = minmax_normalize(cropped_npy)
                np.save(target_filename,cropped_npy)
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
    
    