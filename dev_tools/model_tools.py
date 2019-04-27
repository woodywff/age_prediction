import matplotlib.pyplot as plt
import numpy as np
import os
from dev_tools.my_tools import *

def draw_training_proc(return_list_path):
    '''
    This will draw the change of MSE and person correlation coefficient values during the training process.
    for both training and validation dataset.
    return_list_path: .npy file in which the return_list is saved
    '''
    training_steps,training_losses,training_acces,val_steps,val_losses,\
        val_acces = np.load(return_list_path)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.plot(training_steps, np.log(training_losses),label='training')
    ax1.plot(val_steps,np.log(val_losses),label='validation')
    ax1.set_title('MSE')
    ax1.legend(fontsize=12)
    ax1.set_xlabel('step',fontsize=12)
    ax1.set_ylabel('MSE (natrual log)',fontsize=12)
    
    ax2.plot(training_steps, training_acces,label='training')
    ax2.plot(val_steps,val_acces,label='validation')
    ax2.set_title('Person correlation coefficien')
    ax2.legend(fontsize=12)
    ax2.set_xlabel('step',fontsize=12)
    ax2.set_ylabel('Person correlation coefficient',fontsize=12)
    
    ax1.grid(True)
    ax2.grid(True)
    
    model_name = return_list_path.split('_')[-1].split('.')[0]
    plt.savefig('./img/training_proc_'+model_name+'.pdf', bbox_inches='tight')
    plt.savefig('./img/training_proc_'+model_name+'.png', bbox_inches='tight')
    
    return

def draw_person_corr(pred_age,chro_age,mse,person_corr,test_mae,title='Test Data',save_filename='person_corr'):
    '''
    This is used for test process.
    to draw the correlation between chronological age and predicted age
    '''
    fig = plt.figure()
    plt.title(title)
    plt.xlabel('Chronological Age')
    plt.ylabel('Brain Age (Predicted)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.text(10, 82, 'RMSE = %.2f\nMAE = %.2f\nPerson = %.2f' %(np.sqrt(mse),test_mae,person_corr),
                bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.scatter(chro_age.reshape(-1), pred_age.reshape(-1), c = 'blue',s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('./img/'+save_filename+'.pdf', bbox_inches='tight')
    plt.savefig('./img/'+save_filename+'.png', bbox_inches='tight')
    plt.show()
    return

def person_corr(predicts,labels):
    '''
    (get_acc())
    we are not sure the shape of predicts,labels, (?,1) or (?,),
    so we reshape them into (?,) first.
    '''
    return np.corrcoef(np.vstack((predicts.reshape(-1),labels.reshape(-1))))[0,1]

def tf_calc_mae(predicts,labels):
    '''
    for tensorflow
    we are not sure the shape of predicts,labels, (?,1) or (?,),
    so we reshape them into (?,) first.
    '''
    with tf.name_scope('mae'):
        predict_batches = tf.reshape(predict_batches,[-1,1])
        label_batches = tf.reshape(label_batches,[-1,1])
        mae = tf.reduce_mean(tf.abs(predict_batches - label_batches))
    return mae

def calc_mae(predicts,labels):
    '''
    for ndarray
    we are not sure the shape of predicts,labels, (?,1) or (?,),
    so we reshape them into (?,) first.
    '''
    n = labels.shape[0]
    return np.sum(np.abs(predicts.reshape(-1)-labels.reshape(-1)))/n