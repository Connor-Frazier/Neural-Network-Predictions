import pandas as pd
import csv

#Quandl
import quandl
import time

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

tickers = ["ACN", "ADBE", "ATHN", "ORCL", "WDAY"]

allDataSets = []
for tkr in tickers:
    allDataSets.append(quandl.get_table('SHARADAR/SF1',
                  dimension="ARQ",
                  qopts={
                     "columns": ['fcf']},
                  ticker=tkr))

a = 0
for i in allDataSets:
	TRAIN_SPLIT = len(i) - 1
	data_mean = i[:TRAIN_SPLIT].mean(axis=0)
	data_std = i[:TRAIN_SPLIT].std(axis=0)
	i = (i-data_mean)/data_std
	print(i)
	i.to_csv('fcf_' + tickers[a] + '.csv')
	a+=1
	time.sleep(3)
