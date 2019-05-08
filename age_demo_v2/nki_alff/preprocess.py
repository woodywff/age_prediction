'''
Dataset: NKI
10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)


'''

import sys
sys.path.append("../..")
from dev_tools.my_tools import *
from dev_tools.preprocess_tools import *

# for preprocess_1 resampling
# DESIRED_SHAPE_origin=(176,256,256)
# DESIRED_SHAPE_resample=(110,130,130)
DESIRED_SHAPE = (67,67,67)


def gen_phenotypics(xls_path_name):
    '''
    *** This needs to be modified for different programs ***
    This is an once for all thing.
    To generate phenotypics.csv
    
    xls_path_name: excel file with phenotypic information.
    '''
#     pdb.set_trace()
    if os.path.exists('./phenotypics.csv'):
        print('phenotypics.csv exists already.')
        return
    # get id and age from .xls
    phenotypic_table = xlrd.open_workbook(xls_path_name,'rb')
    pt = phenotypic_table.sheets()[0]

    id_list = pt.col_values(0)
    age_list = pt.col_values(1)

    # delete empty items:
    for i in range(len(id_list)-1,0-1,-1):
        if age_list[i] == '':
            del id_list[i]
            del age_list[i]
        else:
            id_list[i] = int(id_list[i].split('A')[1])

    # shuffle and save the phenotypic info:
    id_list, age_list = get_shuffled(id_list,age_list)
    data_to_save = pd.DataFrame({'id':id_list,'age':age_list})
    data_to_save.to_csv(os.path.join('./','phenotypics.csv'), index=False, sep=',')
    print('phenotypics.csv created.')
    return




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
        re_result = re.match('^ALFFMap_sub-A.*\.npy$',filename)
        if re_result:
            target_filename = os.path.join(target_dir_origin,str(int(filename.split('-A')[1].split('.')[0])))
            if not os.path.exists(target_filename + '.npy'):
#                 cropped_npy = inner_preprocess_1(os.path.join(source_dir,filename))
#                 cropped_npy = minmax_normalize(cropped_npy)
#                 np.save(target_filename,cropped_npy)
                shutil.copyfile(os.path.join(source_dir,filename),target_filename + '.npy')
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
    
    # preprocess_1: .nii to .npy   
    preprocess_1(source_dir,target_dir_origin)
    # preprocess_2: subtract mean values
    preprocess_2(target_dir_origin,target_dir_mean)
    return



def preprocess_main(mean_version=False):
    '''
    mean_version: to use './data_npy/mean_subtracted/' or './data_npy/origin/'
    '''
    print_sep('preprocessing starts')
    # get phenotypics.csv
    gen_phenotypics('/media/woody/Elements/Steve_age_data/participants.xlsx')
    # get training.csv test.csv
    gen_training_test_csv()
    # get preprocessed .npy files
    source_dir = '/media/woody/Elements/Steve_age_data/ALFF_FunRawARWS'
    target_dir = './data_npy'
    gen_npy(source_dir,target_dir)
    # get .tfrecords ready
    if mean_version:
        gen_tfrecord('./training.csv',npy_dir='./data_npy/mean_subtracted/',tf_filename='training_data.tfrec')
        gen_tfrecord('./test.csv',npy_dir='./data_npy/mean_subtracted/',tf_filename='test_data.tfrec')
    else:
        gen_tfrecord('./training.csv',npy_dir='./data_npy/origin/',tf_filename='training_data.tfrec')
        gen_tfrecord('./test.csv',npy_dir='./data_npy/origin/',tf_filename='test_data.tfrec')
    
    my_mkdir('./img')
    my_mkdir('./log')
    print_sep('preprocessing ends')
    return
    


if __name__ == '__main__':
    preprocess_main()
    
    