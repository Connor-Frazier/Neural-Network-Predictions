from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential, Graph



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
BATCH_SIZE = 1
TIME_STEPS = 1


lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, train_data.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True,     kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=lr)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

for x, y in dataset.take(1):
    print(lstm_model.predict(x).shape)



