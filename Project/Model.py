from CTCModel import CTCModel as CTCModel
from tensorflow.keras.layers import Dense, Input,TimeDistributed, Bidirectional, LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Masking
from tensorflow.keras import regularizers

def model (units=500,
            nb_features=129,
            nb_labels=28):
    '''
    model function : to define the architecture of the network used where we have a layer of masking followed by 5 layers of bidirectional LSTM,
    and regularization on the last layer. we added a timedistributed layer before applying CTCmodel.
    return : a network to implement
    ''' 

    #input layer
    x_input = Input((None,nb_features))
    #Masking layer
    x = Masking(mask_value=0)(x_input)
    print(x.shape)
    #1st bidirectional LSTM layer
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    print(x.shape)
    #2nd bidirectional LSTM layer
    x = Bidirectional(LSTM(units,return_sequences=True))(x)
    print(x.shape)
    #3rd bidirectional LSTM layer
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    print(x.shape)
    #4th bidirectional LSTM layer
    x = Bidirectional(LSTM(units,return_sequences=True))(x)
    print(x.shape)
    #5th bidirectional LSTM layer
    x = Bidirectional(LSTM(units,dropout=0.4,kernel_regularizer=regularizers.l2(1e-6),recurrent_regularizer=regularizers.l2(1e-6),return_sequences=True))(x)
    print(x.shape)
    #timedistributed layer
    y_pred = TimeDistributed(Dense(units=nb_labels, activation='softmax'), name='softmax')(x)
    print(y_pred.shape)
    #CTCmodel
    model = CTCModel ([x_input], [y_pred])

    return model
