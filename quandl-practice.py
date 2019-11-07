import quandl

quandl.ApiConfig.api_key = "c41SJX7-N-p3yWF2Ksmk"

data = quandl.get_table('SHARADAR/SF1', ticker='AAPL', dimension='ARQ', calendardate={'gte':'2013-09-01','lte':'2020-01-10'}, paginate=True)

print(data)
print(type(data))

