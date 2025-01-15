"""
    Copyright (C) 2023 Khandaker Foysal Haque
    contact: haque.k@northeastern.edu
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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        condition = tf.cast(tf.reduce_sum(group_norm) > 0.03, tf.float16) # change according to the number of selected group

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






def ConvLayer_group(input, filters, kernelsize, group_lambda,  l1_lambda, group_size, total_group, strides=(2, 2), padding='same', activation='relu', bn=False, name=None):
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
    
    x = layers.Conv2D(filters,kernelsize,strides=strides,padding=padding,name=name, kernel_regularizer=group_lasso(group_list, l1_lambda, group_lambda))(input)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    
    return x

def ConvLayer(x_in, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', bn=False, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
        print("Here")
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x



# def ConvLayer(input, filters, kernelsize):
    
#     x = layers.Conv2D(filters,kernelsize,padding='same')(input)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
    
#     return x

def EmbeddingNet(input_sh, group_lambda, l1_lambda, group_size, total_group):

    input_sh = tf.keras.Input(shape=input_sh)
    print(input_sh.shape)
    x= ConvLayer_group(input_sh, 64, (2, 2), group_lambda,l1_lambda, group_size, total_group,strides=(1, 1), padding='valid', name='1st' + 'group_conv2_1_res_a')
    x = ConvLayer(x,64,3)
    x = layers.MaxPool2D()(x)
    x = ConvLayer(x,32,3)
    x = layers.MaxPool2D()(x)
    x = ConvLayer(x,32,3)
    x = layers.MaxPool2D()(x)
    x = ConvLayer(x,16,3)
    x = layers.MaxPool2D()(x)
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
  

    return keras.Model(inputs=input_sh,outputs=x)

def Decoder(inputshape,output_sh):
    
    x_input = keras.Input(shape=inputshape)
    x = tf.keras.layers.Dense(output_sh, activation=None, name='dense2')(x_input)
    model = tf.keras.Model(inputs=x_input, outputs=x, name='csi_model')
    return model


def csi_network_inc_res(input_sh, output_sh, group_lambda, l1_lambda, group_size, total_group):
    
    embedding_model = EmbeddingNet(input_sh, group_lambda, l1_lambda, group_size, total_group)
    
    decoder_model = Decoder(embedding_model.output.shape[1:], output_sh)
    
    # Pass the encoder's output through the decoder
    full_output = decoder_model(embedding_model.output)
    
    # Create the combined model
    full_model = tf.keras.Model(inputs=embedding_model.input, outputs=full_output)
    
    return full_model







# class Embed_Dec(keras.Model):

#     def __init__(self,inputshape, nclasses):
#         super (Embed_Dec,self).__init__()
#         self.embedding = EmbeddingNet(inputshape)
#         self.decoder = Decoder(self.embedding.layers[-1].output.shape[1:],nclasses)

#     def compile(self,optimizer,loss):
#         super(Embed_Dec,self).compile()
#         self.optimizer = optimizer
#         self.loss = loss
#         self.loss_metric = keras.metrics.CategoricalCrossentropy(name="loss")
#         self.acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")

#     @property
#     def metrics(self):
#         return[self.loss_metric, self.acc_metric]
        
#     # custom fit
#     def train_step(self, data):
#         input, targ = data
#         with tf.GradientTape() as tape: 
#             embed_output = self.embedding(input,training=True)
#             prediction = self.decoder(embed_output,training=True)
#             loss_value = self.loss(y_true=targ,y_pred=prediction)
        
#         variables = self.trainable_variables
#         grad = tape.gradient(loss_value,variables)
#         self.optimizer.apply_gradients(zip(grad,variables))
#         self.loss_metric.update_state(targ,prediction)
#         self.acc_metric.update_state(targ,prediction)
#         return {"loss":self.loss_metric.result(), "accuracy":self.acc_metric.result()}

#     def test_step(self, data):
#         input, targ = data
#         embed_output = self.embedding(input,training=False)
#         prediction = self.decoder(embed_output,training=False)
#         self.loss_metric.update_state(targ,prediction)
#         self.acc_metric.update_state(targ,prediction)
#         return {"loss":self.loss_metric.result(), "accuracy":self.acc_metric.result()}

# class customCallback(keras.callbacks.Callback):
    """
    a custom callback for training Embed_Dec
    """
    def __init__(self,dir,patience):
        super(keras.callbacks.Callback, self).__init__()
        self.checkpoint_path = dir # path to store best model
        self.patience = patience # patience for early stopping
    
    def on_train_begin(self,logs=None):
        self.count = 0 # count for non-best epoch
        self.best_score = np.Inf

    def on_epoch_end(self,epoch,logs=None):
        current = logs.get("val_loss")
        if np.less(current,self.best_score):
            self.best_score = current
            self.count = 0
            self.model.embedding.save(self.checkpoint_path) # save the best model
        else:
            self.count += 1
            if self.count >= self.patience:
                self.model.stop_training = True