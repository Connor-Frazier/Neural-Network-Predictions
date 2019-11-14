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

#cleaning out timestamps
data.pop('calendardate')
data.pop('datekey')
data.pop('reportperiod')
data.pop('lastupdated')

#Create, a train test split, testing if the model can predict the most recent quarter
#This may need more
TRAIN_SPLIT = 23

#Removing emtpty data columns
data = data.select_dtypes(exclude=['object'])

#Normalizing
data_mean = data[:TRAIN_SPLIT].mean(axis=0)
data_std = data[:TRAIN_SPLIT].std(axis=0)
#zscore calculation across data matrix
data = (data-data_mean)/data_std

#Creating data points and labels
train_data = []
train_labels = []
#Data points at the moment consist of a history of 4 qvalue data points and the 5th as the target
for i in range(21):
	train_data.append(data.loc[i:i+3, :].values)
	#'accoci':'assetsnc' are example columns, will be an array for the 4/5 values we choose
	train_labels.append(data.loc[i+4, 'accoci':'assetsnc'].values)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

#creating tensor with features and target values
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

#Showing a couple training data points and labels as examples.
for feat, targ in dataset.take(2):
	print ('Features: {}, Target: {}'.format(feat, targ))

#Dataset is now a tensor where each data point is made up of 4 data points 
#and the next point as the label

#BELOW, unfinished, model creation and testing is next
#The tensor structure may need to be adjusted for the needs of the model
dataset = dataset.cache().shuffle(24).batch(4).repeat()	

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



