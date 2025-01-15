#nohup python RL_model.py ./doppler_traces_ofdm/ S1a40_merge 320 256 1 30 tempsingle_ant A,B,C,D,F 40 8 S2a40_merge,S4a40_merge &

import warnings
import argparse
import cmath
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import os
from dataset_utility import create_dataset_single, expand_antennas, expand_antennas_test
from network_utility_simwisense import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.utils import custom_object_scope
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
import gym
import time
import math
import itertools

class HyperparamTuningEnv(gym.Env):
    def __init__(self, l1_lambda_values, group_lambda_values, antenna_combinations):
        
        self.observation_noise = 0.1 # observation to ensure partial obervation
        
        self.l1_lambda_values = l1_lambda_values # 
        self.group_lambda_values = group_lambda_values
        self.antenna_combinations = antenna_combinations
        
        self.true_state = self.initialize_state()
        self.current_step = 0

        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Discrete(len(self.l1_lambda_values)),
            gym.spaces.Discrete(len(self.group_lambda_values)),
            gym.spaces.Discrete(len(self.antenna_combinations))
            ])
        
        self.action_space_agent1=gym.spaces.Discrete(len(self.antenna_combinations))
        self.action_space_agent2= gym.spaces.Discrete(len(self.l1_lambda_values)*len(self.group_lambda_values))
    
    def initialize_state(self):
        return np.array([
            np.random.choice(len(self.l1_lambda_values)),  # Random index for l1_lambda_values
            np.random.choice(len(self.group_lambda_values)),  # Random index for group_lambda_values
            np.random.choice(len(self.antenna_combinations))  # Random index for antenna_combinations
        ])

    def step(self, action_agent1, action_agent2):
        # Apply the actions (set hyperparameters)
        # print(action_agent1)
        antenna_combination = action_agent1
        l1_lambda = self.l1_lambda_values[action_agent2 // len(self.group_lambda_values)]
        group_lambda = self.group_lambda_values[action_agent2 % len(self.group_lambda_values)]
        self.true_state =l1_lambda, group_lambda, antenna_combination
        # print(self.true_state)

        observation, performance,selected_groups, num_antennas, num_groups  = self.generate_observation(self.true_state)
        
        # calculate the testing accuracy. 
        # reward = self.calculate_reward(self.true_state)
        selected_groups=selected_groups.numpy().tolist()
        reward = (100*performance) - 10*((num_groups-selected_groups.count(0.0))/num_groups)-2*num_antennas
        # Check if the episode is done (based on your criteria)
        self.current_step += 1
        done = self.is_done(observation)
        return observation, reward, done, selected_groups, num_antennas,l1_lambda,group_lambda, performance
    
    
    def calculate_reward_single(self, true_state):
        return np.random.rand(0,1)
    
    def calculate_reward(self, true_state):

        l1_index, group_index,antenna_indices = true_state
        print(l1_index, group_index,antenna_indices)
        antenna=self.antenna_combinations[antenna_indices]
        antenna_str = ','.join(map(str, antenna))

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('dir', help='Directory of data')
        parser.add_argument('subdirs', help='Subdirs for training')
        parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int) # 2002
        parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int) # 340
        parser.add_argument('channels', help='Number of channels', type=int) # 1
        parser.add_argument('batch_size', help='Number of samples in a batch', type=int) # 32
        # parser.add_argument('num_tot', help='Number of antenna * number of spatial streams', type=int) # 4
        parser.add_argument('name_base', help='Name base for the files') # single_ant
        parser.add_argument('activities', help='Activities to be considered') # A,B,C,D,E
        parser.add_argument('group_size', help='group_size', type=int)
        parser.add_argument('total_group', help='total_group', type=int)
        parser.add_argument('subdirs_test', help='Subdirs for testing')
        parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                                '(default 80)', default=80, required=False, type=int) # default 
        parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz ' 
                                            '(default 1)', default=1, required=False, type=int) # default 1 
        args = parser.parse_args()

        antennas_ind =antenna_str  # 1,2,3,4
        antennas_index = [int(num) for num in antennas_ind.split(',')]
       
        bandwidth = args.bandwidth  # 80
        sub_band = args.sub_band  # 1

        group_lambda = group_index
        l1_lambda = l1_index

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
        numbers = antenna_str.split(',')
        num_antennas =  len(antennas_index)
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
        
        name_cache_test = name_base + '_' + str(csi_act) + '_cache_test'
        dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                                stream_ant_test, input_network, batch_size,
                                                shuffle=False, cache_file=name_cache_test, prefetch=True, repeat=True)
        temp_name = subdirs_training.split('_')
        name_model = './saved_model/'+ str(group_lambda)+ '_' + str(l1_lambda) + '_' + subdirs_training + '_' + str(num_antennas) + '_ant' + '_' + str(csi_act) + '_' + antennas_ind + '_network.h5'
        print(name_model)

        csi_model = csi_network_inc_res(input_network, output_shape, group_lambda, l1_lambda, args.group_size, args.total_group)
        optimiz = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
        #save_weights_only=True
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')
        callback_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                        monitor='val_sparse_categorical_accuracy')
    
        csi_model.compile(optimizer=optimiz, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        num_samples_train = len(file_train_selected_expanded) # training data sampled
        num_samples_val = len(file_val_selected_expanded) # validation data sampled
        num_samples_test = len(file_test_selected_expanded) # test data 
        lab, count = np.unique(labels_train_selected_expanded, return_counts=True)
        lab_val, count_val = np.unique(labels_val_selected_expanded, return_counts=True)
        lab_test, count_test = np.unique(labels_test_selected_expanded, return_counts=True)
        train_steps_per_epoch = int(np.ceil(num_samples_train/batch_size))
        val_steps_per_epoch = int(np.ceil(num_samples_val/batch_size))
        test_steps_per_epoch = int(np.ceil(num_samples_test/batch_size))
        start_ime=time.time()
        #####################################training############################################3
        results = csi_model.fit(dataset_csi_train, epochs=15, steps_per_epoch=train_steps_per_epoch,
                                validation_data=dataset_csi_val, validation_steps=val_steps_per_epoch,
                                callbacks=[callback_save,callback_stop])
        end_ime=time.time()
        training_time_minutes=(start_ime-end_ime)/60

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

        selected_groups1 = group_lasso_1(group_list, l1_lambda, group_lambda)(x)
        print("Here is your selected group using group lasso_1 ",selected_groups1.numpy())
        ####################################################################################################
        
        train_labels_true = np.array(labels_train_selected_expanded)

        name_cache_train_test = name_base + '_' + str(csi_act) + '_cache_train_test'
        dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                    stream_ant_train, input_network, batch_size,
                                                    shuffle=False, cache_file=name_cache_train_test, prefetch=False)
        train_prediction_list = csi_model.predict(dataset_csi_train_test,
                                                steps=train_steps_per_epoch)[:train_labels_true.shape[0]]

        train_labels_pred = np.argmax(train_prediction_list, axis=1)

        conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)
        train_accuracy = accuracy_score(train_labels_true, train_labels_pred)
        print("Here is train accuracy", train_accuracy)


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

        metrics_matrix_dict = {
                        'train_accuracy': train_accuracy,
                        'conf_matrix': conf_matrix,
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
        name_file = './train_output/trained_test_' + str(group_lambda) + ',' + str(l1_lambda) + subdirs_training + '_' + str(csi_act) + '_' + antennas_ind + suffix
        with open(name_file, "wb") as fp:  # Pickling
            pickle.dump(metrics_matrix_dict, fp)
        #### testing ###########################
            
        parser.add_argument('subdirs_test', help='Subdirs for testing')
        name_base = args.name_base
        if os.path.exists(name_base + '_' + str(csi_act) + '_cache_complete.data-00000-of-00001'):
            os.remove(name_base + '_' + str(csi_act) + '_cache_complete.data-00000-of-00001')
            os.remove(name_base + '_' + str(csi_act) + '_cache_complete.index') 
        subdirs_complete_test = args.subdirs_test
        labels_complete = []
        all_files_complete = []
        sample_length = args.sample_length
        feature_length = args.feature_length
        channels = args.channels
        input_shape = (num_antennas, sample_length, feature_length, channels)
        input_network = (sample_length, feature_length, channels)
        batch_size = args.batch_size
        output_shape = activities.shape[0]
        labels_considered = np.arange(output_shape)
        activities = activities[labels_considered] 

        for sdir in subdirs_complete_test.split(','):
            exp_save_dir = args.dir + sdir + '/'
            dir_complete = args.dir + sdir + '/complete_antennas_' + str(csi_act) + '/'
            name_labels = args.dir + sdir + '/labels_complete_antennas_' + str(csi_act) + suffix
            with open(name_labels, "rb") as fp:  # Unpickling
                labels_complete.extend(pickle.load(fp))
            name_f = args.dir + sdir + '/files_complete_antennas_' + str(csi_act) + suffix
            with open(name_f, "rb") as fp:  # Unpickling
                all_files_complete.extend(pickle.load(fp))

        file_complete_selected = [all_files_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                              labels_considered]
        labels_complete_selected = [labels_complete[idx] for idx in range(len(labels_complete)) if labels_complete[idx] in
                                    labels_considered]

        file_complete_selected_expanded, labels_complete_selected_expanded, stream_ant_complete = \
            expand_antennas_test(file_complete_selected, labels_complete_selected, num_antennas)

        dataset_csi_complete = create_dataset_single(file_complete_selected_expanded, labels_complete_selected_expanded,
                                                    stream_ant_complete, input_network, batch_size, shuffle=False,
                                                    cache_file=name_base + '_' + str(csi_act) + '_cache_complete')
        num_samples_complete = len(file_complete_selected_expanded)
        lab_complete, count_complete = np.unique(labels_complete_selected_expanded, return_counts=True)
        complete_steps_per_epoch = int(np.ceil(num_samples_complete / batch_size)) 
        complete_labels_true = np.array(labels_complete_selected_expanded)
        start = time.time()
        complete_prediction_list = csi_model.predict(dataset_csi_complete,
                                                    steps=complete_steps_per_epoch)[:complete_labels_true.shape[0]]
        end = time.time()
        timeTaken = end - start

        complete_labels_pred = np.argmax(complete_prediction_list, axis=1)

        conf_matrix = confusion_matrix(complete_labels_true, complete_labels_pred, labels=labels_considered)
        precision, recall, fscore, _ = precision_recall_fscore_support(complete_labels_true,
                                                                    complete_labels_pred,
                                                                    labels=labels_considered)
        accuracy = accuracy_score(complete_labels_true, complete_labels_pred)
        print(subdirs_complete_test,": ", accuracy)
        # merge antennas
        labels_true_merge = np.array(labels_complete_selected)
        pred_max_merge = np.zeros_like(labels_complete_selected)
        for i_lab in range(len(labels_complete_selected)):
            pred_antennas = complete_prediction_list[i_lab*num_antennas:(i_lab+1)*num_antennas, :]
            sum_pred = np.sum(pred_antennas, axis=0)
            lab_merge_max = np.argmax(sum_pred)

            pred_max_antennas = complete_labels_pred[i_lab*num_antennas:(i_lab+1)*num_antennas]
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
            precision_recall_fscore_support(labels_true_merge, pred_max_merge, labels=labels_considered)
        accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

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
                            'timeTaken (in minutes)': timeTaken / 60
                            }

        # name_file = './outputs/complete_different_' + str(csi_act) + '_' + subdirs_complete + '_band_' + str(bandwidth) \
        #             + '_subband_' + str(sub_band) + suffix
        name_file= './test_outputs/test_complete_antennas_' +str(group_lambda)+'_'+str(l1_lambda)+'_'+str(num_antennas)+ '_'+antennas_ind+'_'+ str(csi_act)+'_'+subdirs_complete_test+'.txt'
        with open(name_file, "wb") as fp:  # Pickling
            pickle.dump(metrics_matrix_dict, fp)
        print('accuracy', accuracy_max_merge)
        print('fscore', fscore_max_merge) 
       
        return accuracy_max_merge, selected_groups1, num_antennas, num_groups
    


    
    def generate_observation(self, true_state):
        # Extract the current hyper parameter. 
        
        l1_index, group_index,antenna_indices = true_state
        antenna=self.antenna_combinations[antenna_indices]
        antenna_str = ','.join(map(str, antenna))
        performance_metric, selected_groups, num_antennas, num_groups = self.calculate_reward(true_state)
        # print("performance_metric: ", performance_metric)# Placeholder for actual model performance
        noisy_performance_metric = performance_metric + np.random.normal(0, self.observation_noise)

        # Construct the observation as a list
        observation = [l1_index, group_index, antenna_indices, noisy_performance_metric]

        return observation, performance_metric, selected_groups, num_antennas, num_groups

    def is_done(self, observation):
        # Extract the necessary components from the state
        # For example, if your state includes a performance metric:
        l1_index = observation[0]
        group_index=observation[1]
        antenna_index= observation[2]
        performance_metric= observation[3]
        
        # Criterion 1: Performance Threshold
        # Define a performance threshold. If the performance metric exceeds this threshold, the episode is done.
        # performance_threshold = 0.95  # Adjust based on your specific task
        # if performance_metric >= performance_threshold:
        #     return True

        # Criterion 2: Maximum Steps
        # If the environment tracks the number of steps (you'll need to implement this), end the episode after a certain number.
        max_steps = 15  # Adjust based on your specific task
        if self.current_step >= max_steps:
            return True

        # Criterion 3: Convergence (Optional)
        # If the hyperparameters haven't changed significantly over a certain number of steps, you might consider the episode done.
        # You'll need to track changes in hyperparameters over steps to implement this.

        # If none of the above criteria are met, the episode is not done.
        return False
    
class POMDPAgent:
    def __init__(self, action_space,total_states, l1_lambda_values, group_lambda_values, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.99):
        # Initialize agent properties
        self.action_space = action_space
        self.total_states = total_states
        # print(self.total_states)
        self.l1_lambda_values=l1_lambda_values
        self.group_lambda_values= group_lambda_values
        self.belief_state = np.ones(self.total_states) / self.total_states
       
        # print(self.belief_state)# Initialize belief_state
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        
      
        
        # Initialize the Q-table: Dimensions are [belief_space, action_space]
        # self.q_table = np.zeros((self.belief_space.n, self.action_space.n)) 
        self.q_table = np.zeros((self.total_states, self.action_space.n))

    def select_action(self, belief_state): # selecting best options. 

        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            action= self.action_space.sample()
            
        else:
            # Exploitation: choose the best action based on current belief state and Q-table
            # Find the index of the belief state with the highest probability
            max_belief_state_index = np.argmax(belief_state)
            # Select the action with the highest Q-value for this belief state
            action = np.argmax(self.q_table[max_belief_state_index])
            self.epsilon *= self.epsilon_decay
            
        return action
    
    def update_belief_state(self, action, observation):
        belief_state_index = self.get_belief_state_index(action, observation[0], observation[1], observation[2])
        index_temp = int(np.round(belief_state_index))
        current_belief = self.belief_state[belief_state_index]
        self.belief_state[belief_state_index] = current_belief + cmath.log10(observation[3].real)
    
    
    def learn(self, belief_state, action, reward, next_belief_state):

        # Find the indices of the current and next belief states with the highest probability
        belief_state_index = np.argmax(belief_state)
        next_belief_state_index = np.argmax(next_belief_state)

        # Compute the target Q-value
        td_target = reward + self.discount_factor * np.max(self.q_table[next_belief_state_index])

        # Update the Q-value using the Q-learning update rule
        self.q_table[belief_state_index, action] += self.learning_rate * (td_target - self.q_table[belief_state_index, action])

        # Optionally, you can decay epsilon here if you're using an epsilon-greedy policy
        # self.epsilon *= self.epsilon_decay
    
    def get_belief_state_index(self, action, l1_index, group_index, antenna_index):
        num_l1_values = len(self.l1_lambda_values)
        num_group_values = len(self.group_lambda_values)
        index = (antenna_index * num_l1_values * num_group_values) + (l1_index * num_group_values) + group_index
        return int(index)
    
    def receive_feedback(self, other_belief_state):
        # Simple averaging of belief states
        other_belief_state_ = np.array(other_belief_state)
        logged_belief_state = [cmath.log10(x).real for x in other_belief_state_]
        self.belief_state = self.belief_state + logged_belief_state
        # self.belief_state = self.belief_state + logged_belief_state
    
    def decode_state(self, state_index):
        num_group_values = len(self.group_lambda_values)
        num_l1_values = len(self.l1_lambda_values)

        antenna_index = state_index // (num_l1_values * num_group_values)
        l1_index = (state_index // num_group_values) % num_l1_values
        group_index = state_index % num_group_values

        return self.l1_lambda_values[l1_index], self.group_lambda_values[group_index], antenna_combinations[antenna_index]
    
 
def generate_combinations(elements):
    combinations = []
    for r in range(1, len(elements) + 1):
        for subset in itertools.combinations(elements, r):
            combinations.append(list(subset))
    return combinations


num_tot = np.arange(1,5)
antenna_combinations = generate_combinations([x - 1 for x in num_tot])
# print(antenna_combinations)


        
l1_lambda_values = [0.01, 0.005, 0.001]
group_lambda_values = [0.01, 0.005, 0.001]
env = HyperparamTuningEnv(l1_lambda_values, group_lambda_values, antenna_combinations)
total_states = len(l1_lambda_values) * len(group_lambda_values) * len(antenna_combinations)

agent1 = POMDPAgent(env.action_space_agent1, total_states, l1_lambda_values, group_lambda_values)
agent2 = POMDPAgent(env.action_space_agent2, total_states, l1_lambda_values, group_lambda_values)


num_episodes = 10
episode_metrics1=[]
episode_metrics2=[]
action2=0
agent1.belief_state = np.ones(total_states) / total_states
agent2.belief_state = np.ones(total_states) / total_states
for episode in range(num_episodes):
    epoch_metrics1=[]
    epoch_metrics2=[]
    step=1
    done = False
    while True:
        
        action1 = agent1.select_action(agent1.belief_state)
        observation1, reward1, done1, selected_groups1, num_antennas1,l1_lambda1,group_lambda1, performance1 = env.step(action1, action2)
        agent1.update_belief_state(action1, observation1)
        agent2.receive_feedback(agent1.belief_state)
        
        agent_1={ "observation1": observation1,
                   "reward1": reward1,
                   "done1": done1,
                   "selected_groups1": selected_groups1,
                   "num_antennas1": num_antennas1,
                   "l1_lambda1": l1_lambda1,
                   "group_lambda1": group_lambda1,
                   "performance1":performance1   
        }
        
        
        
        
        action2 = agent2.select_action(agent2.belief_state)
        observation2, reward2, done2, selected_groups2, num_antennas2,l1_lambda2,group_lambda2, performance2 = env.step(action1, action2)
        agent2.update_belief_state(action2, observation2)
        agent1.receive_feedback(agent2.belief_state)
        # Update belief state based on the observation
        
        agent_2={ "observation12": observation2,
            "reward2": reward2,
            "done2": done2,
            "selected_groups2": selected_groups2,
            "num_antennas2": num_antennas2,
            "l1_lambda2": l1_lambda2,
            "group_lambda2": group_lambda2,
            "performance2":performance2     
        }
        
        
        

        # Agents learn from the feedback
        agent1.learn(agent1.belief_state, action1, reward1, agent1.belief_state)
        agent2.learn(agent2.belief_state, action2, reward2, agent2.belief_state)
        
        epoch_metrics1.append(agent_1)
        epoch_metrics2.append(agent_2)
        
        if step==15:
            break
        step=step+1
    episode_metrics1.append(epoch_metrics1)
    episode_metrics2.append(epoch_metrics2)
    # print(f"Episode {episode + 1} finished")

print("Training completed")
    

best_state_index = np.argmax(agent2.q_table.max(axis=1))
best_l1_lambda, best_group_lambda, best_antenna_config = agent2.decode_state(best_state_index)
print("Best l1 lambda value:", best_l1_lambda)
print("Best group lambda value:", best_group_lambda)
print("Best antenna configuration:", best_antenna_config)

#save rl q tabel 
path_ = '/home/vijay/exposing_csi/Python_code/rl_model/rl_save/'
with open(f'{path_}agent1_qtabel.txt', 'wb') as rel:
    pickle.dump(agent1.q_table, rel)

with open(f'{path_}agent2_qtabel.txt', 'wb') as rel:
    pickle.dump(agent2.q_table, rel)


max_metrics = {"Best_l1_lambda_value": best_l1_lambda,
                        "Best_group_lambda_value": best_group_lambda,
                        "Best_antenna_configuration": best_antenna_config
                        }

with open(f'{path_}best_metric.txt', 'wb') as rel:
    pickle.dump(max_metrics, rel)

with open(f'{path_}episode_metrics1.txt', 'wb') as rel:
    pickle.dump(episode_metrics1, rel)
    
with open(f'{path_}episode_metrics2.txt', 'wb') as rel:
    pickle.dump(episode_metrics2, rel)
    
    
    
    
    

