from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

import time

#Quandl
import quandl

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

tickers = ['AAPL', 'ATVI', 'ATHN', 'MSFT', 'ADBE', 'ORCL', 'CRM', 'WDAY', 'ACN', 'TWTR', 'CNQR', 'PEGA', 'AZPN', 'BYI']

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

	#Removing emtpty data columns
	data = data.select_dtypes(exclude=['object'])
	data = data.loc[:, (data != 0).any(axis=0)]
	#Need to remove rows if they have nan in some of the cells

	#Setting variables
	steps = 4
	featuresCount = 13
	TEST_SPLIT = 5
	TRAIN_SPLIT = len(data) - TEST_SPLIT
	numberOfPredictedFeatures = 4

	#Normalizing
	data_mean = data[:TRAIN_SPLIT].mean(axis=0)
	data_std = data[:TRAIN_SPLIT].std(axis=0)
	data = (data-data_mean)/data_std

	#Setting parameters and creating data points and labels
	train_data = []
	train_labels = []

	#Data points at the moment consist of a history of 4 qvalue data points and the 5th as the target
	for i in range(TRAIN_SPLIT):
		train_data.append(data.iloc[i:i+steps, :].values)
		#'accoci':'assetsnc' are example columns, will be an array for the 4/5 values we choose
		train_labels.append(data.loc[i+steps, 'ebit':'fcf'].values)
	
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)


	#Setting up test data
	test_data = []
	test_labels = []

	for j in range(TEST_SPLIT):
		temp = TRAIN_SPLIT - 1 - steps + j
		test_data.append(data.iloc[temp:temp + steps, :].values)
		test_labels.append(data.loc[TRAIN_SPLIT + j, 'ebit':'fcf'].values)

	test_data = np.array(test_data)
	test_labels = np.array(test_labels)


	train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], featuresCount))
	model = Sequential()
	model.add(LSTM(50, activation='relu', input_shape=(steps, featuresCount)))
	model.add(Dense(4))
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

	model.fit(train_data, train_labels, epochs=200)

	# predicting the test data points
	sse = 0
	for k in range(TEST_SPLIT):
		x_input = test_data[k]
		x_input = x_input.reshape((1, steps, featuresCount))
		output = model.predict(x_input, verbose=0)
		print("Prediction: ")
		print(output)
		print("True: ")
		print(test_labels[k])
		for i in range(numberOfPredictedFeatures):
			sse += (output[0][i] - test_labels[k][i])**2
		print("sse: ")
		print(sse)
 
	#The true way to evaluate(i think but the accuracy is always 0 because the numbers are not exact)
	test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], featuresCount))
	test_loss, test_acc = model.evaluate(test_data,  test_labels)

	print('\nTest accuracy:', test_acc)

	time.sleep(3)


