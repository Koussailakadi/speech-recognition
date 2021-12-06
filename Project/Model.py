from CTCModel import CTCModel as CTCModel
from tensorflow.keras.layers import Dense, Input,TimeDistributed, Bidirectional, LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Masking


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




