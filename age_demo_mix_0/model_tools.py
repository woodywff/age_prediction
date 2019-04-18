import matplotlib.pyplot as plt
import numpy as np
import os

def draw_training_proc(training_steps,training_losses,training_acces,
                       val_steps,val_losses,val_acces,model_name):
    '''
    This will draw the change of MSE and person correlation coefficient values during the training process.
    for both training and validation dataset.
    steps: list
    training_losses,training_acces,val_losses,val_acces: list, acces refers accuracies
    model_name: str, the deep learning model's name
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.plot(training_steps, training_losses)
    ax1.plot(val_steps,val_losses)
    ax1.set_title('MSE')
    ax2.plot(training_steps, training_acces)
    ax2.plot(val_steps,val_acces)
    ax2.set_title('Person correlation coefficien')
    ax1.grid(True)
    ax2.grid(True)
    plt.savefig('./img/training_proc_'+model_name+'.pdf', bbox_inches='tight')
    plt.savefig('./img/training_proc_'+model_name+'.png', bbox_inches='tight')

    plt_data_path_name = './img/training_proc_mse_'+model_name+'_pltdata_' + time_now()
    if not os.path.exists(plt_data_path_name + '.npy'):
        np.save(plt_data_path_name, np.array([training_steps,training_losses,val_steps,val_losses]))
    else:
        print(plt_data_path_name + '.npy exists already.')

    plt_data_path_name = './img/training_proc_person_'+model_name+'_pltdata_' + time_now()
    if not os.path.exists(plt_data_path_name + '.npy'):
        np.save(plt_data_path_name, np.array([training_steps,training_acces,val_steps,val_acces]))
    else:
        print(plt_data_path_name + '.npy exists already.')