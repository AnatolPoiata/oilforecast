import time
from datetime import datetime, timedelta

import pandas as pd

#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pycaret.regression import *

tomorrow_date = datetime.now() + timedelta(days=1)
tomorrow = tomorrow_date.strftime('%Y-%m-%d')
lag_range = 5
data_lag = 250

starting_date = "2024-09-01"
starting_date = datetime.strptime(starting_date, '%Y-%m-%d')

def pycaret_model(df, tr='open'):

    data = df[['date', tr]].copy()
    data.set_index('date', inplace=True)

    for lag in range(1, lag_range+1):
        data[f'Lag_{lag}'] = data[tr].shift(lag)

    data_lagged = data.dropna()
    data_lagged = data_lagged.tail(data_lag)

    data_lagged.sort_index(ascending=False, inplace=True)

    print(data_lagged.head(20))

    nbr_days = (tomorrow_date-starting_date).days

    ff = []

    for d in range(0, nbr_days):
        end_date = starting_date+timedelta(days=d)
        print('-----------------------------------------------------------')
        print('end_date: ',end_date)

        if (end_date.isoweekday()>5):
            print('weekend')
            continue

        historic_data = data_lagged[data_lagged.index<end_date].copy()
        print(historic_data.head())
#        historic_data = data_lagged[data_lagged['date']<end_date].copy()

        s = setup(data = historic_data, target = tr, train_size = 0.8, session_id=123)

        best = compare_models()
#        print('Best:',best)

        model = create_model(best)

    
        predict_holdout = predict_model(model)
        predict_holdout['Error pct'] = (predict_holdout[tr] - predict_holdout['prediction_label'])/predict_holdout[tr]*100
#        print(predict_holdout)
#        print('Avg error:', predict_holdout['Error pct'].mean())

        NewData = data_lagged[data_lagged.index==end_date].copy()
        if (NewData.shape[0]==0):
            continue

        forecasts=predict_model(model, data=NewData)

        print(forecasts)

        ff.append({'date': end_date, tr: forecasts['prediction_label'][0]})

    df = pd.DataFrame(ff)

    return df


if __name__ == "__main__":

    df_history = pd.read_csv('historical_data.csv')
    df_history['date'] = pd.to_datetime(df_history['date'])

    forecast_open = pycaret_model(df_history, 'open')
    print(forecast_open.head())
    print(forecast_open.shape)
    

    forecast_close = pycaret_model(df_history, 'close')

    print(forecast_close.head())
    print(forecast_close.shape)


    df_forecast = pd.merge(forecast_open, forecast_close, on='date', how='left')

    df_test = pd.merge(df_forecast, df_history, on='date', how='left', suffixes=('_forecast', '_history'))

    try:    
        df_old_forecast=pd.read_csv('forecast_test.csv')
    except:
        df_old_forecast = pd.DataFrame()

    df_test=pd.concat([df_old_forecast, df_test], axis=0)
    
    df_test.drop_duplicates(subset=['date'],inplace=True)

    df_test.to_csv('forecast_test.csv', index=False, date_format="%Y-%m-%d")

