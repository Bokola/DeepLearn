import numpy as np
import pandas as pd
# load all time zones
from pytz import all_timezones
from scipy.special import delta

# 7.1 Converting strings to dates
# date stringd
date_strings = np.array(['03-04-2005 11:35 PM',
'23-05-2010 12:01 AM',
'04-09-2009 09:09 PM'])
# convert to datetimes
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]
# you might want to add errors parameter to handle errors
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p', errors="coerce") for date in date_strings]
# 7.2 Handling time zones
# create datetime
pd.Timestamp('2025-05-16 10:16:00', tz = 'Europe/London')
# add tz to previously created time
date = pd.Timestamp('2025-05-16 10:16:00')
date_in_Nairobi = date.tz_localize('Africa/Nairobi')
date_in_Nairobi
# set tz to every date
dates = pd.Series(pd.date_range('5/5/2025', periods=3, freq='M'))
dates.dt.tz_localize('Africa/Nairobi')
# load all time zones
all_timezones[0:10]

# 7.3 Selecting dates and times

df = pd.DataFrame()
df["date"] = pd.date_range('1/1/2025', periods=100, freq="D")
# select observations btwn 2 dates
df[(df["date"] > '2025-1-1') & (df["date"] < '2025-4-30')]

# Alternative is using index
df = df.set_index(['date'])
# select obs btw 2 datetimes
df.loc['2025-1-1':'2025-4-30']

# 7.4 Breaking up date into multiple features

# Create data frame
dataframe = pd.DataFrame()

# Create five dates
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')
# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute
# Show three rows
dataframe.head(3)

# 7.5 calculating the difference between dates

dataframe = pd.DataFrame()
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# calculate duration between features
dataframe['Left'] - dataframe['Arrived']

# remove days output
pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))
# 7.6 Encoding days of week

# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="ME"))
# show days of week
dates.dt.day_name()
# show days of week as numeric

# 7.7 Creating a lagged feature

dataframe = pd.DataFrame()
# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]
# Lagged values by one row
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)
dataframe

# 7.8 Using rolling time windows

# Create datetimes
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# Create data frame, set index
dataframe = pd.DataFrame(index=time_index)
# Create feature
dataframe["Stock_Price"] = [1,2,3,4,5]
# calculate rolling mean
dataframe.rolling(window=2).mean()
# 7.9 Handling missing data in time series
# fot ts data we can interpolate

# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# Create data frame, set index
dataframe = pd.DataFrame(index=time_index)
# Create feature with a gap of missing values
dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]
dataframe.interpolate()

# we can also fill with the latest known value
dataframe.bfill()