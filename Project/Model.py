from CTCModel import CTCModel as CTCModel
from tensorflow.keras.layers import Dense, Input,TimeDistributed, Bidirectional, LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Masking
from tensorflow.keras import layers


def model (units=500,
            nb_features=129,
            nb_labels=28):

    x_input = Input((None,nb_features))
    x = Masking(mask_value=0)(x_input)
    print(x.shape)
    x = Bidirectional(LSTM(units,return_sequences=True))(x)
    print(x.shape)
    x = Bidirectional(LSTM(units,return_sequences=True))(x)
    print(x.shape)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    print(x.shape)
    x = Bidirectional(LSTM(units,return_sequences=True))(x)
    print(x.shape)
    x = Bidirectional(LSTM(units,return_sequences=True))(x)
    print(x.shape)
    y_pred = TimeDistributed(Dense(units=nb_labels, activation='softmax'), name='softmax')(x)
    print(y_pred.shape)
    model = CTCModel ([x_input], [y_pred])

    return model


def model_with_CNN(units=500,
            nb_features=129,
            nb_labels=28): 
        
        # couche de convolution: 
        x_input = Input((None, nb_features), name="input")
        # Expand the dimension to use 2D CNN.
        x = layers.Reshape((-1, nb_features, 1), name="expand_dim")(x_input)
        # Convolution layer 1
        x = layers.Conv2D(filters=32,kernel_size=[11, 41],strides=[1, 2],padding="same",use_bias=False,name="conv_1")(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        # Convolution layer 2
        x = layers.Conv2D(filters=32,kernel_size=[11, 21],strides=[1, 2],padding="same",use_bias=False,name="conv_2")(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

        #______BLSTM layers:
        #x_input = Input((None,nb_features))
        #print(x_input.shape)
        x = Masking(mask_value=0)(x)
        x = Bidirectional(LSTM(units,return_sequences=True))(x)
        x = Bidirectional(LSTM(units,return_sequences=True))(x)
        x = Bidirectional(LSTM(units, return_sequences=True))(x)
        x = Bidirectional(LSTM(units,return_sequences=True))(x)
        x = Bidirectional(LSTM(units,return_sequences=True))(x)
        y_pred = TimeDistributed(Dense(units=nb_labels, activation='softmax'), name='softmax')(x)

        """#_______dense  layer:
        # Dense layer
        x = layers.Dense(units=units, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        # Classification layer
        y_pred = layers.Dense(units=28, activation="softmax", name="dense_2")(x)"""

        model = tf.keras.Model(inputs=x_input, outputs= y_pred)
        model = CTCModel ([x_input], [y_pred])

        return model




