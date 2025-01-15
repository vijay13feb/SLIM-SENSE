
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

import tensorflow as tf

# @tf.keras.utils.register_keras_serializable(package='Custom', name='l4')
class group_lasso_1(tf.keras.regularizers.Regularizer):
  def __init__(self, group_list, group_lambda, l1_lambda):
    self.group_list = group_list
    self.group_lambda = group_lambda
    self.l1_lambda = l1_lambda
    
    

  def __call__(self, x):
    l1_regularization = self.l1_lambda * tf.reduce_sum(tf.abs(x))
    group_regularization = 0
    selected_groups  = [] 

    for group_index, group in enumerate(self.group_list):
        # group_weights = [tf.abs(x[:,i]) for i in group]
        #for conv2d
        group_weights = [tf.abs(x[:, :, :, i]) for i in group]
        # group_norm = tf.sqrt(tf.reduce_sum(tf.stack(group_weights, axis=1), axis=1)) # L2 norm 
        group_norm = tf.reduce_sum(tf.stack(group_weights, axis=1), axis=1) # l1 norm 
        group_regularization += tf.reduce_sum(group_norm)
        condition = tf.cast(tf.reduce_sum(group_norm) > 0.04, tf.float16) # change according to the number of selected group

        # Append group_index to selected_groups if condition is met
        selected_groups.append(condition * tf.constant(group_index+1, dtype=tf.float16))
       
    selected_groups = tf.stack(selected_groups)
    # print("Selected Groups:", selected_groups)
    

    # return l1_regularization + self.group_lambda * group_regularization
    return selected_groups
  
  # def get_config(self):
  #   return {'l2': float(self.group_lambda)}
  def get_config(self):
    config = {
        'group_list': self.group_list,
        'l1_lambda': self.l1_lambda,
        'group_lambda': self.group_lambda
    }
    return config



# @tf.keras.utils.register_keras_serializable(package='Custom', name='l3')
class group_lasso(tf.keras.regularizers.Regularizer):
  def __init__(self, group_list, group_lambda, l1_lambda):
    self.group_list = group_list
    self.group_lambda = group_lambda
    self.l1_lambda = l1_lambda
    
    

  def __call__(self, x):
    l1_regularization = self.l1_lambda * tf.reduce_sum(tf.abs(x))
    group_regularization = 0
    selected_groups  = [] 

    for group_index, group in enumerate(self.group_list):
        # group_weights = [tf.abs(x[:,i]) for i in group]
        #for conv2d
        group_weights = [tf.abs(x[:, :, :, i]) for i in group]
        # group_norm = tf.sqrt(tf.reduce_sum(tf.stack(group_weights, axis=1), axis=1)) # L2 norm 
        group_norm = tf.reduce_sum(tf.stack(group_weights, axis=1), axis=1) # l1 norm 
        group_regularization += tf.reduce_sum(group_norm) + self.group_lambda * tf.reduce_sum(tf.abs(tf.reduce_sum(tf.stack(group_weights, axis=1), axis=1)))
    #     condition = tf.cast(tf.reduce_sum(group_norm) > 0.04, tf.float16) # change according to the number of selected group

    # # Append group_index to selected_groups if condition is met
    #     selected_groups.append(condition * tf.constant(group_index+1, dtype=tf.float16))
       
    # selected_groups = tf.stack(selected_groups)
    # print("Selected Groups:", selected_groups)
    

    return l1_regularization + self.group_lambda * group_regularization
  
  # def get_config(self):
  #   return {'l2': float(self.group_lambda)}
  def get_config(self):
    config = {
        'group_list': self.group_list,
        'l1_lambda': self.l1_lambda,
        'group_lambda': self.group_lambda
    }
    return config



def conv2d_bn(x_in, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', bn=False, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
        print("Here")
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x
    #x2 = conv2d_bn(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')
def conv2d_bn_(x_in, filters, kernel_size,l1_lambda, group_lambda, group_size, total_group, strides=(2, 2), padding='same', activation='relu', bn=False, name=None):
   #conv2d_bn_(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + '1stgroup_conv2_1_res_a', group_lambda=0.1,l1_lambda=0.1)
    group_list = []
    group_size = group_size
    # selected_groups=[]
    
    num_groups = total_group # Number of groups
    total_features = group_size*num_groups
    filters = total_features

    for i in range(num_groups):
        group_start = i * group_size
        group_end = (i + 1) * group_size
        group_list.append(list(range(group_start, group_end)))

    l1_lambda = l1_lambda # L1 regularization strength
    group_lambda = group_lambda

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name,kernel_regularizer=group_lasso(group_list, l1_lambda, group_lambda))(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x


def reduction_a_block_small(x_in,group_lambda, l1_lambda, group_size, total_group,base_name):


    group_lambda=group_lambda
    l1_lambda=l1_lambda
    group_size=group_size
    total_group=total_group
    
    # l1_lambda = 0.01  # L1 regularization strength
    # group_lambda = 0.01
    #x_in= tf.keras.layers.Dense(1480, activation=None)(x_in)
    #dense= tf.keras.layers.Conv2D(filters, kernel_size=(2, 2), strides=(1, 1),activation=None, kernel_regularizer=group_lasso(group_list, l1_lambda, group_lambda))(x_in)

    x_in = conv2d_bn_(x_in, 5, (2, 2), group_lambda,l1_lambda, group_size, total_group,strides=(2, 2), padding='valid', name=base_name + 'group_conv2_1_res_a')
    
    x1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(x_in)

    x2 = conv2d_bn(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')

    x3 = conv2d_bn(x_in, 3, (1, 1), name=base_name + 'conv3_1_res_a')
    x3 = conv2d_bn(x3, 6, (2, 2), name=base_name + 'conv3_2_res_a')
    x3 = conv2d_bn(x3, 9, (4, 4), strides=(2, 2), padding='same', name=base_name + 'conv3_3_res_a')

    x4 = tf.keras.layers.Concatenate()([x1, x2, x3])
    return x4


def csi_network_inc_res(input_sh, output_sh, group_lambda, l1_lambda, group_size, total_group):
    x_input = tf.keras.Input(input_sh)

    x2 = reduction_a_block_small(x_input, group_lambda, l1_lambda, group_size, total_group, base_name='1st')

    x3 = conv2d_bn(x2, 3, (1, 1), name='conv4')

    x = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None, name='dense2')(x)
    
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_model')
    
    return model


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


    
    # group_list = []
    # group_size = 40
    # total_features = 2960
    # num_groups = 74  # Number of groups

    # for i in range(num_groups):
    #     group_start = i * group_size
    #     group_end = (i + 1) * group_size
    #     group_list.append(list(range(group_start, group_end)))

    # l1_lambda = 0.01  # L1 regularization strength
    # group_lambda = 0.01
    #densex1= tf.keras.layers.Dense(100,activation=None, kernel_regularizer=L2Regularizer(l2=0.5), name='dense_first')(x_in)
    # densex1= tf.keras.layers.Dense(total_features,activation=None, kernel_regularizer=tf.keras.regularizers.L1(0.01), name='dense_first')(x_in)
    #densex2= tf.keras.layers.Dense(total_features,input_shape=(total_features,), activation=None,kernel_regularizer=group_lasso(group_list, l1_lambda, group_lambda))(x_in)
    # densex2= tf.keras.layers.Dense(10, activation=None,)(densex1)
    
    #densex2= tf.keras.layers.Conv2D(x_in, filters=5, kernel_size=(3, 3), strides=(2, 2),activation=None, kernel_regularizer=group_lasso(group_list, l1_lambda, group_lambda))
        # group_list= []
    # count=0
    # total_features= features
    # group_size = 40 # size of the group 
    # grop_size_ = 40 # just for creating the group equal to group_size
    # ran= total_features//grop_size_
    # for i in range(1, ran+1):
    #   group_list.append([j for j in range(count, grop_size_)])
    #   count=count+group_size
    #   grop_size_ =grop_size_ +group_size