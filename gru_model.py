from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow and tf.keras
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from tensorflow.python.framework import ops
from tensorflow import keras
# from keras.models import Sequential, Graph
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

import time

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

#Quandl
import quandl

import sklearn
import sklearn.preprocessing

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

# tickers = ['AAPL', 'ATVI', 'ATHN', 'MSFT', 'ADBE', 'ORCL', 'CRM', 'WDAY', 'ACN', 'TWTR', 'CNQR', 'PEGA', 'AZPN', 'BYI']
tickers = ['AAPL']

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"



for ticker in tickers:

    data = quandl.get_table('SHARADAR/SF1',
    dimension = "ARQ",
    qopts={"columns":['datekey','revenueusd','assets','capex','debt','ebit','equityusd','ev','fcf','inventory','marketcap','netinc','taxexp','workingcapital']},
    ticker=ticker)

    # Reversing the data so that the oldest timestamp is first
    data = data.iloc[::-1]
    data.pop('datekey')

    # Removing emtpty data columns
    data = data.select_dtypes(exclude=['object'])
    data = data.loc[:, (data != 0).any(axis=0)]
    # Need to remove rows if they have nan in some of the cells

    # Setting variables
    steps = 4
    featuresCount = 13
    TEST_SPLIT = 5
    TRAIN_SPLIT = len(data) - TEST_SPLIT
    numberOfPredictedFeatures = 4

    # Normalizing
    data_mean = data[:TRAIN_SPLIT].mean(axis=0)
    data_std = data[:TRAIN_SPLIT].std(axis=0)
    data = (data - data_mean) / data_std

    sequence_len = 10
    raw_data = data.values
    data = []
    for i in range(len(raw_data) - sequence_len):
        data.append(raw_data[i: i + sequence_len])
    data = np.array(data)
    valid_set_size = int(np.round(10 / 100 * data.shape[0]))
    test_set_size = int(np.round(10 / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]
    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_valid.shape = ", x_valid.shape)
    print("y_valid.shape = ", y_valid.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    n_steps = sequence_len - 1
    n_inputs = 4
    n_neurons = 200
    n_outputs = 4
    n_layers = 2
    learning_ratep = 0.001
    batch_size = 5
    n_epochs = 100
    train_set_size = x_train.shape[0]
    test_set_size = x_test.shape[0]
    tf.reset_default_graph()
    xplaceholder = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    yplaceholder = tf.placeholder(tf.float32, [None, n_outputs])

    layers = tf.contrib.rnn.GRUCell(num_units = neurons, activations=tf.nn.leaky_relu)
            for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_optimizer = optimizer.minimize(loss)
