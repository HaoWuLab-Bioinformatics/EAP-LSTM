import tensorflow
import keras
from keras import backend as K, regularizers
import os
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf
#from keras.engine import Layer
from keras.layers import Layer
from keras import models, initializers
from keras.layers import Input, Lambda, Concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D,UpSampling1D
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers import BatchNormalization, InputLayer, Input, Bidirectional, LSTM
from keras.models import Sequential, Model

def EAP_LSTM_drosophila(data_shape1,data_shape2,kernel_size1,kernel_size2,kernel_size3,kernel_size4,pool_size, dropout_rate, stride):
    input_data1 = Input(shape=data_shape1)
    input_data2 = Input(shape=data_shape2)
    x = Conv1D(256, kernel_size=kernel_size1, strides=stride, padding='same')(input_data1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)


    x = Conv1D(60, kernel_size=kernel_size2, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(60, kernel_size=kernel_size3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(120, kernel_size=kernel_size4, strides=stride,  padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(8, return_sequences=True))(x)
    x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x_f = Conv1D(256, kernel_size=7, strides=stride, padding='same')(input_data2)
    x_f = BatchNormalization()(x_f)
    x_f = Activation('relu')(x_f)
    x_f = MaxPooling1D(pool_size=pool_size)(x_f)
    x_f = Dropout(dropout_rate)(x_f)
    x_f = Flatten()(x_f)

    x_f = Dense(100)(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = Activation('relu')(x_f)
    x_f = Dropout(dropout_rate)(x_f)

    x = Concatenate(axis=1)([x, x_f])
    outputs = []
    tasks = ['Dev', 'Hk']
    for task in tasks:
        outputs.append(Dense(1, activation='linear', name=str('Dense_' + task))(x))
    model = Model([input_data1,input_data2], outputs)
    print(model.summary())
    return model


def EAP_LSTM_human(data_shape1,data_shape2,data_shape3,kernel_size1,kernel_size2,kernel_size3,kernel_size4,pool_size, dropout_rate, stride):
    input_data1 = Input(shape=data_shape1)
    input_data2 = Input(shape=data_shape2)
    input_data3 = Input(shape=data_shape3)

    x = Conv1D(256, kernel_size=kernel_size1, strides=stride, padding='same')(input_data1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(60, kernel_size=kernel_size2, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(60, kernel_size=kernel_size3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(120, kernel_size=kernel_size4, strides=stride,  padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(8, return_sequences=True))(x)
    x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x_signal = Conv1D(256, kernel_size=kernel_size1, strides=stride, padding='same')(input_data3)
    x_signal = BatchNormalization()(x_signal)
    x_signal = Activation('relu')(x_signal)
    x_signal = MaxPooling1D(pool_size=pool_size)(x_signal)
    x_signal = Dropout(dropout_rate)(x_signal)

    x_signal = Conv1D(60, kernel_size=kernel_size2, strides=stride, padding='same')(x_signal)
    x_signal = BatchNormalization()(x_signal)
    x_signal = Activation('relu')(x_signal)
    x_signal = MaxPooling1D(pool_size=pool_size)(x_signal)
    x_signal = Dropout(dropout_rate)(x_signal)

    x_signal = Conv1D(60, kernel_size=kernel_size3, strides=stride, padding='same')(x_signal)
    x_signal = BatchNormalization()(x_signal)
    x_signal = Activation('relu')(x_signal)
    x_signal = MaxPooling1D(pool_size=pool_size)(x_signal)
    x_signal = Dropout(dropout_rate)(x_signal)

    x_signal = Conv1D(120, kernel_size=kernel_size4, strides=stride, padding='same')(x_signal)
    x_signal = BatchNormalization()(x_signal)
    x_signal = Activation('relu')(x_signal)
    x_signal = MaxPooling1D(pool_size=pool_size)(x_signal)
    x_signal = Dropout(dropout_rate)(x_signal)

    x_signal = Bidirectional(LSTM(8, return_sequences=True))(x_signal)
    x_signal = Flatten()(x_signal)

    x_signal = Dense(256)(x_signal)
    x_signal = BatchNormalization()(x_signal)
    x_signal = Activation('relu')(x_signal)
    x_signal = Dropout(dropout_rate)(x_signal)

    x_signal = Dense(256)(x_signal)
    x_signal = BatchNormalization()(x_signal)
    x_signal = Activation('relu')(x_signal)
    x_signal = Dropout(dropout_rate)(x_signal)

    x_f = Conv1D(256, kernel_size=7, strides=stride, padding='same')(input_data2)
    x_f = BatchNormalization()(x_f)
    x_f = Activation('relu')(x_f)
    x_f = MaxPooling1D(pool_size=pool_size)(x_f)
    x_f = Dropout(dropout_rate)(x_f)

    x_f = Flatten()(x_f)

    x_f = Dense(100)(x_f)
    x_f = BatchNormalization()(x_f)
    x_f = Activation('relu')(x_f)
    x_f = Dropout(dropout_rate)(x_f)


    x = Concatenate(axis=1)([x, x_signal, x_f])
    outputs = Dense(1, activation='linear')(x)

    model = Model([input_data1,input_data2,input_data3], outputs)
    print(model.summary())
    return model

