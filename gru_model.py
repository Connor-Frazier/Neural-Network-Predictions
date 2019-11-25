import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import time

import os
import pandas as pd 
import quandl

'''
GRU Model needs at least 1 Dense layer w/ 4 nuerons
Adding more Dense layers has little impact on accuracy
Changing optimizer has larger impact on accuracy
Cannot use GlobalAveragePooling1D or Embedding layers because they take
data in different shape
'''

def gru_model1(train_data, train_labels):
    model = keras.Sequential([
            keras.layers.GRU(50, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(4),
            ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=200)
    return model

def gru_model2(train_data, train_labels):
    model = keras.Sequential([
            keras.layers.GRU(200, activation='relu', input_shape=(steps, featuresCount), return_sequences=False),
            keras.layers.Dense(4),
            keras.layers.Dense(4)
            ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=200)
    return model

def gru_model3(train_data, train_labels):
    model = keras.Sequential([
            keras.layers.GRU(200, activation='relu', input_shape=(steps, featuresCount), return_sequences=False),
            keras.layers.Dense(4),
            ])
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=200)
    return model

def gru_model4(train_data, train_labels):
    model = keras.Sequential([
            keras.layers.GRU(200, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(4),
            keras.layers.Dense(4),
            keras.layers.Dense(4),
            keras.layers.Dense(4),
            ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=200)
    return model

# lowest accuracy model (b/c optimizer is RMSprop)
def gru_model5(train_data, train_labels):
    model = keras.Sequential([
            keras.layers.GRU(200, activation='relu', input_shape=(steps, featuresCount)),
            keras.layers.Dense(4),
            keras.layers.Dense(4),
            ])
    model.compile(optimizer='RMSprop', loss='mse', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=200)
    return model


quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

#tickers = ['AAPL', 'ATVI', 'ATHN', 'MSFT', 'ADBE', 'ORCL', 'CRM', 'WDAY', 'ACN', 'TWTR', 'CNQR', 'PEGA', 'AZPN', 'BYI']

tickers = ['AAPL']

for t in tickers:
    data = quandl.get_table('SHARADAR/SF1', dimension = "ARQ",
                            qopts={"columns":['ticker','datekey','revenueusd','assets','capex','debt','ebit','equityusd','ev','fcf','inventory','marketcap','netinc','taxexp','workingcapital']},
                             ticker = t, paginate = True)
    data = data.iloc[::-1]
    data.pop('datekey')
    
    data = data.select_dtypes(exclude=['object'])
    data = data.loc[:, (data != 0).any(axis=0)]
    
    steps = 4
    featuresCount = 13
    TEST_SPLIT = 5
    TRAIN_SPLIT = len(data) - TEST_SPLIT
    numberOfPredictedFeatures = 4
    
    data_mean = data[:TRAIN_SPLIT].mean(axis=0)
    data_std = data[:TRAIN_SPLIT].std(axis=0)
    data = (data-data_mean)/data_std
    
    train_data = []
    train_labels = []
    for i in range(TRAIN_SPLIT):
        train_data.append(data.iloc[i:i+steps, :].values)
        train_labels.append(data.loc[i+steps, 'ebit':'fcf'].values)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    
    test_data = []
    test_labels = []
    for j in range(TEST_SPLIT):
    	temp = TRAIN_SPLIT - 1 - steps + j
    	test_data.append(data.iloc[temp:temp + steps, :].values)
    	test_labels.append(data.loc[TRAIN_SPLIT + j, 'ebit':'fcf'].values)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], featuresCount))

    # Change which model you're using here
    model = gru_model1(train_data, train_labels)

    sse = 0
    for k in range(TEST_SPLIT):
        x_input = test_data[k]
        x_input = x_input.reshape((1, steps, featuresCount))
        output = model.predict(x_input, verbose=0)
        print("Prediction: ", output)
        print("Actual: ", test_labels[k])
        for i in range(numberOfPredictedFeatures):
            sse += (output[0][i] - test_labels[k][i])**2
        print("sse: ")
        print(sse)

    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], featuresCount))
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    print('\nTest loss:', test_loss)
    print('Test accuracy:', test_acc)
    
    time.sleep(5)
