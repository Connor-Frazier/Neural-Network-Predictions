from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

#Quandl
import quandl

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

data = quandl.get_table('SHARADAR/SF1', ticker='AAPL', dimension='ARQ', calendardate={'gte':'2013-09-01','lte':'2020-01-10'}, paginate=True)
print(type(data))

# target = data.pop('workingcapital')
# print(type(target))

#cleaning out timestamps
data.pop('calendardate')
data.pop('datekey')
data.pop('reportperiod')
data.pop('lastupdated')

TRAIN_SPLIT = 23

#Removing emtpty data columns
data = data.select_dtypes(exclude=['object'])
#Normalizing
data_mean = data[:TRAIN_SPLIT].mean(axis=0)
data_std = data[:TRAIN_SPLIT].std(axis=0)

data = (data-data_mean)/data_std

#Creating data points and labels
train_data = []
train_labels = []


for i in range(21):
	train_data.append(data.loc[i:i+3, :].values)
	train_labels.append(data.loc[i+4, 'accoci':'assetsnc'].values)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

#creating tensor with features and target values
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

for feat, targ in dataset.take(2):
	print ('Features: {}, Target: {}'.format(feat, targ))

dataset = dataset.cache().shuffle(24).batch(4).repeat()	

# for feat, targ in dataset.take(2):
# 	print ('Features: {}, Target: {}'.format(feat, targ))

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=train_data.shape[-2:]),
    tf.keras.layers.Dense(1)
])


simple_lstm_model.compile(optimizer='adam', loss='mae')

for x, y in dataset.take(1):
    print(simple_lstm_model.predict(x).shape)



