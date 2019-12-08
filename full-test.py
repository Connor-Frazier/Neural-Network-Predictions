from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

import time

import csv

#Quandl
import quandl

def getModel(model_name):
	print("Running " + model_name)
	if model_name == 'gru1':
		return keras.Sequential([
            keras.layers.GRU(5, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(4)
            ])
	elif model_name == 'gru2':
		return keras.Sequential([
            keras.layers.GRU(10, activation='relu', input_shape=(steps, featuresCount), return_sequences=False),
            keras.layers.Dense(10),
            keras.layers.Dense(10),
            keras.layers.Dense(4)
            ])
	elif model_name == 'gru3':
		return keras.Sequential([
            keras.layers.GRU(30, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(20),
            keras.layers.Dense(10),
            keras.layers.Dense(4)
            ])
	elif model_name == 'lstm1':
		return keras.Sequential([
            keras.layers.LSTM(5, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(4)
            ])
	elif model_name == 'lstm2':
		return keras.Sequential([
            keras.layers.LSTM(10, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(4)
            ])
	elif model_name == 'lstm3':
		return keras.Sequential([
            keras.layers.LSTM(30, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(4)
            ])	



#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

tickers = ['AAPL', 'ATVI', 'ATHN', 'MSFT', 'ADBE', 'ORCL', 'CRM', 'WDAY', 'ACN', 'TWTR', 'CNQR', 'PEGA', 'AZPN', 'BYI']

models = ["gru1", "gru2", "gru3", "lstm1", "lstm2", "lstm3"]
# models = ["gru1"] this is used for script testing
results = {}
for modelName in models:
	results[modelName] = {}

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

	#Cleaning data if necessary
	data = data.select_dtypes(exclude=['object'])
	data = data.loc[:, (data != 0).any(axis=0)]

	#Setting variables
	steps = 4
	featuresCount = 13
	TEST_SPLIT = 4
	TRAIN_SPLIT = len(data) - TEST_SPLIT
	numberOfPredictedFeatures = 4
	epochs = 100
	predictions = ['ebit', 'ev', 'netinc', 'revenueusd']

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
		#'accoci':'assetsnc' are example columns, will be an array for the 4
		train_labels.append(data.loc[i+steps, predictions].values)
	
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)

	#Fix feature coutn in case data is different than expected
	featuresCount = len(train_data[1][0]);


	#Setting up test data
	test_data = []
	test_labels = []

	for j in range(TEST_SPLIT):
		temp = TRAIN_SPLIT - 1 - steps + j
		test_data.append(data.iloc[temp:temp + steps, :].values)
		test_labels.append(data.loc[TRAIN_SPLIT + j, predictions].values)

	test_data = np.array(test_data)
	test_labels = np.array(test_labels)

	for modelName in models:
		#Get the model to test
		model = getModel(modelName)

		#Compile and train
		# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
		model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mse'])
		model.fit(train_data, train_labels, epochs=epochs)
		
		# predicting the test data points
		print("Testing")
		results[modelName][ticker] = {}
		sse = 0
		for k in range(TEST_SPLIT):
			x_input = test_data[k]
			x_input = x_input.reshape((1, steps, featuresCount))
			output = model.predict(x_input, verbose=0)
			for i in range(numberOfPredictedFeatures):
				sse += (output[0][i] - test_labels[k][i])**2
		print("sse: ")
		print(sse)
		results[modelName][ticker] = sse	
 
# Save results for each model in a csv file
with open('sse_results.csv', 'w', newline='') as csvfile:
	fieldnames = ['model'] + tickers
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

	writer.writeheader()
	for key, value in results.items():
		row = value
		row['model'] = key
		writer.writerow(row)


