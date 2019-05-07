'''
Replication of Ruyi's code.
Dataset: IXI
10% for test, 90% for training. (Option: k-folds cross validation, not implemented yet.)

# after tar IXI dataset
# rename files
# get phenotypics.csv
# get training.csv test.csv
# get preprocessed .npy files. details of preprocess please refer to the functions.

'''
import sys
sys.path.append("..")
from dev_tools.my_tools import *
from dev_tools.preprocess_tools import *

DESIRED_SHAPE = (67,67,67)

#-------------------- once for a life thing (dataset specific)-------------------------------

def IXI_rename():
    '''
    This is an once for all thing.
    rename .nii.gz files as IXI[ixi_id].nii.gz
    '''
    STAMP = False # if any file has the right name for Regular Expression

    target_dir = '/media/woody/Elements/age_data/IXI/IXI-T1'
    files_list = os.listdir(target_dir)
    
    pbar = ProgressBar().start()
    n_bar = len(files_list)
    
    for i,filename in enumerate(files_list):
        re_result = re.match('^IXI.*-.*-.*-T1\.nii\.gz$',filename)
        if re_result:
            STAMP = True
            new_filename = filename.split('-')[0] + '.nii.gz'
            os.rename(os.path.join(target_dir,filename),os.path.join(target_dir,new_filename))
        pbar.update(int(i*100/(n_bar-1)))
    pbar.finish()
    
    if not STAMP:
        print('IXI_rename() finished. No file found.')
    else:
        print('IXI_rename() finished. Done.')
    return

def gen_phenotypics():
    '''
    This is an once for all thing.
    to generate phenotypics.csv
    '''
    if os.path.exists('./phenotypics.csv'):
        print('phenotypics.csv exists already.')
        return
    # get id and age from .xls
    phenotypic_table = xlrd.open_workbook('IXI.xls','rb')
    pt = phenotypic_table.sheets()[0]

    id_list = pt.col_values(0)[1:]
    index_age = np.where(np.array(pt.row_values(0))=='AGE')[0][0]
    age_list = pt.col_values(index_age)[1:]

    # delete empty items:
    for i in range(len(id_list)-1,0-1,-1):
        if age_list[i] == '':
            del id_list[i]
            del age_list[i]

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
    
    source_dir = '/media/woody/Elements/age_data/IXI/IXI-T1'
    target_dir_origin = './data_npy/origin'
    '''
    nii_list = os.listdir(source_dir)
     
    pbar = ProgressBar().start()
    n_bar = len(nii_list)
    
    for i,filename in enumerate(nii_list):
        re_result = re.match('^IXI[0-9]*\.nii\.gz$',filename)
        if re_result:
            target_filename = os.path.join(target_dir_origin,str(int(filename.split('IXI')[1].split('.')[0])))
            if not os.path.exists(target_filename + '.npy'):
                cropped_npy = inner_preprocess_1(os.path.join(source_dir,filename)，rot_ixi=True)
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
    print_sep('preprocessing starts')
    # after tar IXI dataset
    # rename files
    IXI_rename()
    # get phenotypics.csv
    gen_phenotypics()
    # get training.csv test.csv
    gen_training_test_csv()
    # get preprocessed .npy files
    # please refer to the comments of gen_npy()
    source_dir = '/media/woody/Elements/age_data/IXI/IXI-T1'
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

