from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Quandl
import quandl

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

data = quandl.get_table('SHARADAR/SF1', ticker='AAPL', dimension='ARQ', calendardate={'gte':'2013-09-01','lte':'2020-01-10'}, paginate=True)

target = data.pop('workingcapital')

#Removing emtpty data columns
data = data.select_dtypes(exclude=['object'])

#cleaning out timestamps
data.pop('calendardate')
data.pop('datekey')
data.pop('reportperiod')
data.pop('lastupdated')

#creating tensor with features and target values
#This needs to change to the time series set up
dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values))

for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))