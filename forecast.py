import time
from datetime import datetime, timedelta

import pandas as pd

#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from pycaret.regression import *


import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


tomorrow_date = datetime.now() + timedelta(days=1)
tomorrow = tomorrow_date.strftime('%Y-%m-%d')
lag_range = 5
data_lag = 250

def pycaret_model(df, tr='open'):

    print(df.columns)
    print(df.head())

    data = df[['date', tr]].copy()
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    data.set_index('date', inplace=True)

    for lag in range(1, lag_range+1):
        data[f'Lag_{lag}'] = data[tr].shift(lag)

    data_lagged = data.dropna()

    data_lagged = data_lagged.tail(data_lag)

    historic_data = data_lagged

    s = setup(data = historic_data, target = tr, train_size = 0.8, session_id=123)

    best = compare_models()
    print('Best:',best)

    model = create_model(best)

    predict_holdout = predict_model(model)
    predict_holdout['Error pct'] = (predict_holdout[tr] - predict_holdout['prediction_label'])/predict_holdout[tr]*100
    print(predict_holdout)
    print('Avg error:', predict_holdout['Error pct'].mean())

    NewData = data_lagged.tail(1)

    record = {}

    for lag in range(1, lag_range+1):
        record[f'Lag_{lag}'] = data_lagged[tr].tail(lag)[0]

    NewData = pd.DataFrame(record, index=[datetime.strptime(tomorrow, '%Y-%m-%d')])
    NewData.index.name='date'

    forecasts=predict_model(model, data=NewData)

    return forecasts

def send_to_mail(date_tomorrow, forecast_open, forecast_close):




    subject = 'Crude Oil Price Forecast  -  '+ date_tomorrow
    body = subject + '\n'+'Crude Oil Price Forecast - ' + date_tomorrow 

    message = 'Subject: {}\n\n{}'.format(subject, body).encode("utf8")
    text = 'Open:  '+str(forecast_open)+'\n'+'Close:  '+str(forecast_close)

    msg = MIMEMultipart("alternative")
#    msg = MIMEMultipart()
    msg['From'] = 'report.monitoring@viascope.md'
    msg['Subject'] = subject

# convert both parts to MIMEText objects and add them to the MIMEMultipart message
    part1 = MIMEText(text, "plain")
    msg.attach(part1)

    email_list = ['anatol.poiata@ac-tech.com', 'nebojsa33@gmail.com', 'vitalie.tataru@ac-tech.com', 'vitalie.ursachi@ac-tech.com']

    for email in email_list:

        msg['To'] = email

        try:
            server = smtplib.SMTP('mail.viascope.md', 587)
            server.ehlo()
            server.starttls()
            server.ehlo()   
            server.login('report.monitoring@viascope.md', 'qp8y8k8PP[')

            email_response = server.sendmail(msg['From'], email, msg.as_string())

#               server.sendmail(msg['From'], to_a, msg.as_string())
            print('Email sent to ', email)
            print('Result', email_response)


            server.close()
            print('Emails sent !')

            send_result = 1


        except Exception as e:
            print('Cannot send the result email...', e)
            send_result = 0

    return send_result


if __name__ == "__main__":

    df_history = pd.read_csv('historical_data.csv')

    print(df_history.columns)
    print(df_history.head())

    forecast_open = pycaret_model(df_history, 'open')
    forecast_close = pycaret_model(df_history, 'close')

    forecast_open = forecast_open['prediction_label'][0]
    forecast_close = forecast_close['prediction_label'][0]

    new_row = {'date': tomorrow, 'open': forecast_open, 'close': forecast_close}
    print(new_row)

    try:
        df_forecast = pd.read_csv('forecast_data.csv')
        df_forecast.iloc[-1] = new_row
    except:
        df_forecast = pd.DataFrame([{'date':tomorrow, 'open':forecast_open, 'close':forecast_close}])

    df_forecast.drop_duplicates(keep='last', inplace=True)

    df_forecast.to_csv('forecast_data.csv', index=False, date_format="%Y-%m-%d")

    print('date:', tomorrow ,'Prediction:',' Open: ',forecast_open, ' Close: ',forecast_close)
    send_result = send_to_mail(tomorrow, forecast_open, forecast_close)

    print('send_result=', send_result)

