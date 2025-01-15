

"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import warnings
import argparse
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf
from dataset_utility import create_dataset_single, expand_antennas, expand_antennas_test
from network_utility import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.utils import custom_object_scope
import sys
import os
import time
# import loggingconda 
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress specific Python warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow-specific logs
# logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for training')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int) # 2002
    parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int) # 340
    parser.add_argument('channels', help='Number of channels', type=int) # 1
    parser.add_argument('batch_size', help='Number of samples in a batch', type=int) # 32
    parser.add_argument('num_tot', help='Number of antenna * number of spatial streams', type=int) # 4
    parser.add_argument('name_base', help='Name base for the files') # single_ant
    parser.add_argument('activities', help='Activities to be considered') # A,B,C,D,E
    parser.add_argument('group_lambda', help='Group Lambda value of group regulariser', type=float)
    parser.add_argument('l1_lambda', help='l1 Lambda value of L1 regulariser', type=float)
    parser.add_argument('group_size', help='group_size', type=int)
    parser.add_argument('total_group', help='total_group', type=int)
    parser.add_argument('antennas_index', help='antennas_index')
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int) # default 
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz ' 
                                           '(default 1)', default=1, required=False, type=int) # default 1 
    args = parser.parse_args()

    antennas_ind = args.antennas_index # 1,2,3,4
    antennas_index = [int(num) for num in antennas_ind.split(',')]
    
    
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)
    
    # For GPU efficiency --------------------------------------------------------------
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)
    # --------------------------------------------------------------------

    bandwidth = args.bandwidth  # 80
    sub_band = args.sub_band  # 1
    group_lambda = args.group_lambda
    l1_lambda = args.l1_lambda

    csi_act = args.activities  # A,B,C,D,E
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities) # list of activities. 

    name_base = args.name_base # single_ant
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_train.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_train.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_train.index')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_val.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_val.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_val.index')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_train_test.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_train_test.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_train_test.index')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_test.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) + '_cache_test.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_test.index')

    subdirs_training = args.subdirs  # like S1a
    labels_train = [] 
    all_files_train = []
    labels_val = []
    all_files_val = []
    labels_test = []
    all_files_test = []
    sample_length = args.sample_length # 340
    feature_length = args.feature_length # 2002
    channels = args.channels # 1
    num_antennas = args.num_tot # 4
    #input shape for the model  training num of antennas X sample length X feature length X channels. 
    input_shape = (num_antennas, sample_length, feature_length, channels) # 4 X 340 X 2002 X 1.
    # input network sample length 340, featur length 2002 and channels 1. 
    input_network = (sample_length, feature_length, channels) # 340 X 100 X 1.
    #batch size is 32. 
    batch_size = args.batch_size
    # ouput shapes is number of activities to be detected. 
    output_shape = activities.shape[0]
    # labels considered. 
    labels_considered = np.arange(output_shape)
    # activities
    activities = activities[labels_considered]

    suffix = '.txt'

    for sdir in subdirs_training.split(','):
        exp_save_dir = args.dir + sdir + '/'
        dir_train = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/' # all trains files
        # ./doppler_traces/S1a/train_antennas_A,B,C,D,F/

        name_labels = args.dir + sdir + '/labels_train_antennas_' + str(csi_act) + suffix # labels train antennas 
        #  ./doppler_traces/S1a/labels_train_antennas_A,B,C,D,F.txt
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_train.extend(pickle.load(fp))

        name_f = args.dir + sdir + '/files_train_antennas_' + str(csi_act) + suffix

        with open(name_f, "rb") as fp:  # Unpickling
            all_files_train.extend(pickle.load(fp))

        dir_val = args.dir + sdir + '/val_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_val_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_val.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_val_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_val.extend(pickle.load(fp))

        dir_test = args.dir + sdir + '/test_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + '/labels_test_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_test.extend(pickle.load(fp))
        name_f = args.dir + sdir + '/files_test_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_test.extend(pickle.load(fp))
    
    

    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                           labels_considered]
  
    # all files like 0.txt, 1.txt, 2.txt etc. 
    
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                             labels_considered]
 
   
    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected, labels_train_selected, num_antennas,antennas_index)
    # list all files 4 times for each antennas. 
    
    
    name_cache = name_base+ subdirs_training+ '_' + str(csi_act) + '_cache_train'
    dataset_csi_train = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                              stream_ant_train, input_network, batch_size,
                                              shuffle=True, cache_file=name_cache, prefetch=True, repeat=True)
 
    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                         labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                           labels_considered]

    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas, antennas_index)

    name_cache_val = name_base + '_' + str(csi_act) + '_cache_val'
    dataset_csi_val = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                            stream_ant_val, input_network, batch_size,
                                            shuffle=False, cache_file=name_cache_val, prefetch=True, repeat=True)
    
    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                          labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                            labels_considered]

    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas, antennas_index)
   
    # file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
    #     expand_antennas_test(file_test_selected, labels_test_selected, num_antennas)
   
    name_cache_test = name_base + '_' + str(csi_act) + '_cache_test'
    dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                             stream_ant_test, input_network, batch_size,
                                             shuffle=False, cache_file=name_cache_test, prefetch=True, repeat=True)
    # iterator = iter(dataset_csi_train)

# Access the first element (you can iterate through the dataset similarly).

# Now, you can access the shape of the first element.
    # sys.exit(0)
# print(dataset_csi_test[0].shape).
# print(dataset_csi_val[0].shape).
# csi model takes input shape and output shape.
    
    temp_name = subdirs_training.split('_')
    #change the intial name of the directory here 
    name_model = './model_50/'+ str(group_lambda)+ '_' + str(l1_lambda) + '_' + subdirs_training + '_' + str(num_antennas) + "_"+str(name_base) + '_' + str(csi_act) + '_' + antennas_ind +'_' + str(batch_size)+ '_network.keras'
    print(name_model)
    # name = name_model = name_base +subdirs_training+ '_' + str(csi_act) + '_network.h5'
    # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    # tf.keras.mixed_precision.experimental.set_policy(policy)    
    csi_model = csi_network_inc_res(input_network, output_shape, group_lambda, l1_lambda, args.group_size, args.total_group)
    optimiz = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
    #save_weights_only=True
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')
    callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                    monitor='val_sparse_categorical_accuracy')
  
    csi_model.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    #csi_model.summary()
    
    num_samples_train = len(file_train_selected_expanded) # training data sampled
    num_samples_val = len(file_val_selected_expanded) # validation data sampled
    num_samples_test = len(file_test_selected_expanded) # test data 
    lab, count = np.unique(labels_train_selected_expanded, return_counts=True)
    lab_val, count_val = np.unique(labels_val_selected_expanded, return_counts=True)
    lab_test, count_test = np.unique(labels_test_selected_expanded, return_counts=True)
    train_steps_per_epoch = int(np.ceil(num_samples_train/batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val/batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test/batch_size))
    
    #################################################################################################################################################
    # model training
    #record time taken
    start = time.time()
    results = csi_model.fit(dataset_csi_train, epochs=10, steps_per_epoch=train_steps_per_epoch,
                            validation_data=dataset_csi_val, validation_steps=val_steps_per_epoch,
                            callbacks=[callback_save,callback_stop])
    
    csi_model.save(name_model)
    end = time.time()
    
    timeTaken = end - start #gives the time taken in seconds
    #################################################################################################################################################
    with custom_object_scope({'group_lasso': group_lasso}):
        csi_model = tf.keras.models.load_model(name_model)

    # csi_model = tf.keras.models.load_model(name_model)
    conv_layer = csi_model.get_layer('1stgroup_conv2_1_res_a')
    # # kernel_weights = conv_layer.kernel
    # print(conv_layer)
    x = conv_layer.kernel

    group_list = []
    selected_groups=[]
    group_size = args.group_size
    num_groups = args.total_group  # Number of groups
    
    total_features = group_size*num_groups
    filters = total_features

    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group_list.append(list(range(group_start, group_end)))
    # here yopu have chnaged the values and values must be same to network utility
    l1_lambda = l1_lambda # L1 regularization strength
    group_lambda = group_lambda
  
    selected_groups = group_lasso(group_list, l1_lambda, group_lambda)(x)
    # print("Here is your selected group using group lasso ",selected_groups.numpy())

    selected_groups = group_lasso_1(group_list, l1_lambda, group_lambda)(x)
    print("Here is your selected group using group lasso_1 ",selected_groups.numpy())

    # selected_group_values = csi_model.select_group.numpy()
    # print(selected_groups)
    # config = csi_model.get_config()
    # print(config['layers'][1]['config']['kernel_regularizer']['config']['select_group'])
###########################################################################################################

    train_labels_true = np.array(labels_train_selected_expanded)

    name_cache_train_test = name_base + '_' + str(csi_act) + '_cache_train_test'
    dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                   stream_ant_train, input_network, batch_size,
                                                   shuffle=False, cache_file=name_cache_train_test, prefetch=False)
    train_prediction_list = csi_model.predict(dataset_csi_train_test,
                                              steps=train_steps_per_epoch)[:train_labels_true.shape[0]]

    train_labels_pred = np.argmax(train_prediction_list, axis=1)

    conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)
    accuracy = accuracy_score(train_labels_true, train_labels_pred)
    print("Here is train accuracy", accuracy)


    # val
    val_labels_true = np.array(labels_val_selected_expanded)
    val_prediction_list = csi_model.predict(dataset_csi_val, steps=val_steps_per_epoch)[:val_labels_true.shape[0]]

    val_labels_pred = np.argmax(val_prediction_list, axis=1)

    conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)
    accuracy = accuracy_score(val_labels_true, val_labels_pred)
    print("Here is val accuracy", accuracy)

    # test
    test_labels_true = np.array(labels_test_selected_expanded)

    test_prediction_list = csi_model.predict(dataset_csi_test, steps=test_steps_per_epoch)[
                            :test_labels_true.shape[0]]

    test_labels_pred = np.argmax(test_prediction_list, axis=1)

    conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_true,
                                                                   test_labels_pred,
                                                                   labels=labels_considered, zero_division=1)
    accuracy = accuracy_score(test_labels_true, test_labels_pred)
    print("Here is test accuracy", accuracy)

    # merge antennas test
    labels_true_merge = np.array(labels_test_selected)
    pred_max_merge = np.zeros_like(labels_test_selected)
    for i_lab in range(len(labels_test_selected)):
        pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
        lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

        pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
        lab_max_merge = -1
        if lab_unique.shape[0] > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]
            if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:  # ex aequo between two labels
                lab_max_merge = lab_merge_max
            else:
                lab_max_merge = lab_unique_sort[0]
        else:
            lab_max_merge = lab_unique[0]
        pred_max_merge[i_lab] = lab_max_merge

    conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels=labels_considered)
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
        precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered, zero_division=1)
    accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)
    print('accuracy_max_merge', accuracy_max_merge)
    metrics_matrix_dict = {'conf_matrix': conf_matrix,
                           'accuracy_single': accuracy,
                           'precision_single': precision,
                           'recall_single': recall,
                           'fscore_single': fscore,
                           'conf_matrix_max_merge': conf_matrix_max_merge,
                           'accuracy_max_merge': accuracy_max_merge,
                           'precision_max_merge': precision_max_merge,
                           'recall_max_merge': recall_max_merge,
                           'fscore_max_merge': fscore_max_merge,
                           'selected_groups':selected_groups.numpy(),
                           'timeTaken (in seconds)': timeTaken,
                           }

    name_file = './model_50/' + str(group_lambda)+','+str(l1_lambda)+ subdirs_training+'_'+str(csi_act) + '_' +antennas_ind+ suffix
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(metrics_matrix_dict, fp)

    # impact of the number of antennas
    # if num_antennas==4: 
    #     print("impact of number of antennas")
    #     one_antenna = [[0], [1], [2], [3]]
    #     two_antennas = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    #     three_antennas = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    #     four_antennas = [[0, 1, 2, 3]]
    #     seq_ant_list = [one_antenna, two_antennas, three_antennas, four_antennas]
    #     average_accuracy_change_num_ant = np.zeros((num_antennas,))
    #     average_fscore_change_num_ant = np.zeros((num_antennas,))
    #     labels_true_merge = np.array(labels_test_selected)
    #     for ant_n in range(num_antennas):
            
    #         seq_ant = seq_ant_list[ant_n]
    #         # print(seq_ant)
    #         num_seq = len(seq_ant)
    #         print("here", num_seq)
    #         for seq_n in range(num_seq):
    #             pred_max_merge = np.zeros((len(labels_test_selected),))
    #             ants_selected = seq_ant[seq_n]
    #             for i_lab in range(len(labels_test_selected)):
    #                 pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
    #                 pred_antennas = pred_antennas[ants_selected, :]

    #                 lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

    #                 pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
    #                 pred_max_antennas = pred_max_antennas[ants_selected]
    #                 lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
    #                 lab_max_merge = -1
    #                 if lab_unique.shape[0] > 1:
    #                     count_argsort = np.flip(np.argsort(count))
    #                     count_sort = count[count_argsort]
    #                     lab_unique_sort = lab_unique[count_argsort]
    #                     if count_sort[0] == count_sort[1] or lab_unique.shape[0] > ant_n - 1:  # ex aequo between two labels
    #                         lab_max_merge = lab_merge_max
    #                     else:
    #                         lab_max_merge = lab_unique_sort[0]
    #                 else:
    #                     lab_max_merge = lab_unique[0]
    #                 pred_max_merge[i_lab] = lab_max_merge

    #             _, _, fscore_max_merge, _ = precision_recall_fscore_support(labels_true_merge, pred_max_merge,
    #                                                                         labels=[0, 1, 2, 3, 4], zero_division=1)
    #             accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)
    #             print(accuracy_max_merge)

    #             average_accuracy_change_num_ant[ant_n] += accuracy_max_merge
    #             average_fscore_change_num_ant[ant_n] += np.mean(fscore_max_merge)

    #         average_accuracy_change_num_ant[ant_n] = average_accuracy_change_num_ant[ant_n] / num_seq
    #         average_fscore_change_num_ant[ant_n] = average_fscore_change_num_ant[ant_n] / num_seq

    #     metrics_matrix_dict = {'average_accuracy_change_num_ant': average_accuracy_change_num_ant,
    #                         'average_fscore_change_num_ant': average_fscore_change_num_ant}

    #     name_file = './outputs_10/trained_change_number_antennas_test_' + str(group_lambda)+','+str(l1_lambda)+ subdirs_training+'_'+str(csi_act) + '_' +antennas_ind+'.txt'
    #     with open(name_file, "wb") as fp:  # Pickling
    #         pickle.dump(metrics_matrix_dict, fp)


    # print(csi_model.layers)
    # x = csi_model.layers[1].kernel  # Get the kernel weights of the first dense layer
    # group_list= []
    # count=0
    # total_features= 2960
    # group_size = 40 # size of the group 
    # grop_size_ = 40 # just for creating the group equal to group_size
    # ran= total_features//grop_size_
    # for i in range(1, ran+1):
    #   group_list.append([j for j in range(count, grop_size_)])
    #   count=count+group_size
    #   grop_size_ =grop_size_ +group_size
    
    # l1_lambda = 0.01  # L1 regularization strength
    # group_lambda = 0.01
    # selected_groups = group_lasso(group_list, l1_lambda, group_lambda)(x)
    # print("Here is your selected group",selected_groups)
##################################################################################

  # class CustomCallback(tf.keras.callbacks.Callback):
    #     def on_epoch_begin(self, epoch, logs=None):
    #         print(f"Epoch {epoch + 1}/{self.params['epochs']}")
    #         print("Custom Regularizer - Beginning of Epoch")