import pandas as pd

#Quandl
import quandl
import time

#Getting data
quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

tickers = ['AAPL', 'ATVI', 'ATHN', 'MSFT', 'ADBE', 'ORCL', 'CRM', 'WDAY', 'ACN', 'TWTR', 'CNQR', 'PEGA', 'AZPN', 'BYI']

allDataSets = []
for tkr in tickers:
 allDataSets.append(quandl.get_table('SHARADAR/SF1',
              dimension="ARQ",
              qopts={
                 "columns": ['revenueusd', 'assets', 'capex', 'ebit', 'equityusd',
                          'ev', 'fcf', 'marketcap', 'netinc', 'taxexp', 'workingcapital']},
              ticker=tkr))


for i in allDataSets:
	TRAIN_SPLIT = len(i) - 1
	data_mean = i[:TRAIN_SPLIT].mean(axis=0)
	data_std = i[:TRAIN_SPLIT].std(axis=0)
	i = (i-data_mean)/data_std

	print(i)
	time.sleep(3)


# data = quandl.get_table('SHARADAR/SF1',
#                          dimension = "ARQ",
#                          qopts={"columns":['datekey','revenueusd','assets','capex','debt','ebit','equityusd','ev','fcf','inventory','marketcap','netinc','taxexp','workingcapital']},
#                          ticker='AAPL')
# #cleaning out timestamps
# # data.pop('calendardate')
# print(data)