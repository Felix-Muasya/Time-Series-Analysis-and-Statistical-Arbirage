"""
Here I explore the prophet forecasting library.
Checking the how long it takes to run with datetime.
"""


import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

# introduce our data
raw_data = pd.read_csv("../BTC.csv",index_col=[0], parse_dates=[0])


# my interest is in the close price, so I'll choose that
data = raw_data.iloc[:,[3]]


# print(data)

# plot it to get a general overview
color_pal = sns.color_palette()  # make a color palette
data.plot(style='.', figsize=(10, 5), ms=1, color=color_pal[0],
          title="BTC Close")
#plt.show()


# Create some Time Series Features to visualize our data and better understand it

category_type = CategoricalDtype(categories=['Monday', 'Tuesday', 'Wednesday',
                                             'Thurday', 'Friday', 'Saturday',
                                             'Sunday'],
                                 ordered=True)


def create_features(data, label=None):
    data_copy = data.copy()
    data_copy['Date'] = data_copy.index
    data_copy['DayOfWeek'] = data_copy['Date'].dt.dayofweek
    data_copy['Weekday'] = data_copy['Date'].dt.day_name()
    data_copy['Weekday'] = data_copy['Weekday'].astype(category_type)
    data_copy['Quarter'] = data_copy['Date'].dt.quarter
    data_copy['Month'] = data_copy['Date'].dt.month
    data_copy['Year'] = data_copy['Date'].dt.year
    data_copy['DayOfYear'] = data_copy['Date'].dt.dayofyear
    data_copy['WeekOfYear'] = data_copy['Date'].dt.isocalendar().week
    data_copy['Date_Offset'] = (data_copy.Date.dt.month*100 + data_copy.Date.dt.day-320) % 1300

    data_copy['Season'] = (data_copy.Date.dt.month*100 + data_copy.Date.dt.day - 320) % 1300

    data_copy['Season'] = pd.cut(data_copy['Date_Offset'], [0, 300, 602, 900, 1300],
                                 labels=['Spring', 'Summer', 'Fall', 'Winter'])

    X = data_copy[['DayOfWeek', 'Quarter', 'Month', 'Year', 'DayOfYear', 'WeekOfYear', 'Weekday', 'Season']]

    if label:
        y = data_copy[label]
        return X, y
    return X


X, y = create_features(data, label='close')

features_n_target = pd.concat([X, y], axis=1)
#print(features_n_target)



# Plotting our features

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=features_n_target.dropna(),
            x='Weekday',
            y='close',
            hue='Season',
            ax=ax,
            linewidth=1)
ax.set_title('CLose Price by Day of week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Price USD')
ax.legend(bbox_to_anchor=(1, 1))
#plt.show()



#Train / test split

split_date = '2022-04-01'
data_train = data.loc[data.index <= split_date].copy()
data_test = data.loc[data.index > split_date].copy()

# visualize the split and rename some columns
data_test \
    .rename(columns={'close':'Test Set'}) \
    .join(data_train.rename(columns={'close':'Training set'}),
          how='outer') \
    .plot(figsize=(15, 5), title='Price', style='.')
#plt.show()


# Going into the prophet model. timestamp column renamed to ds, close renamed to y

data_train_prophet = data_train.reset_index() \
    .rename(columns= {"Date" : "ds",
                      'close':'y'})

# Predicting the future
model = Prophet()
model.fit(data_train_prophet)

data_test_prophet = data_test.reset_index() \
    .rename(columns={'Date':"ds",
                     'close':'y'})
data_test_fcst = model.predict(data_test_prophet)
# print(data_test_fcst.to_string())

# printing out the prediction
fig, ax = plt.subplots(figsize=(10, 5))
fig = model.plot(data_test_fcst, ax=ax)
#plt.show()

# comparing prediction to real data
f, ax = plt.subplots(figsize=(10, 5))
ax.scatter(data_test.index, data_test['close'], color='r')
fig = model.plot(data_test_fcst, ax=ax)
plt.show()

# TODO: finish up forecasting and comparison to actual values. Post analysis