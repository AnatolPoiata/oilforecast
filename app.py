import pandas as pd
import numpy as np

import streamlit as st
#from streamlit_extras.app_logo import add_logo

import altair as alt

import datetime as dt
from datetime import datetime, date, timedelta

# from pycaret.regression import *


st.header("Crude Oil Price Foreacst")

st.markdown(
	 """
<style>
span[data-baseweb="tag"]  {
  background-color: blue  !important;
}
</style>
""",
	 unsafe_allow_html=True,
)


column_config_h={
	"date": st.column_config.DateColumn(
		'Date',
		format=None,
		),
	'open_forecast': st.column_config.NumberColumn(
		"Open Forecast",
		format="%.6f",
	),
	'open_history': st.column_config.NumberColumn(
		"Open History",
		format="%.6f",
	),
	'close_forecast': st.column_config.NumberColumn(
		"Open Forecast",
		format="%.6f",
	),
	'close_history': st.column_config.NumberColumn(
		"Open History",
		format="%.6f",
	),
	'open_pct': st.column_config.NumberColumn(
		'error %',
		format="%.2f",
	),
	'close_pct': st.column_config.NumberColumn(
		'error %',
		format="%.2f",
	)
}



df_history = pd.read_csv('historical_data.csv')
df_history['date'] = pd.to_datetime(df_history['date'], format='%Y-%m-%d').dt.date

df_forecast = pd.read_csv('forecast_data.csv')
df_forecast['date'] = pd.to_datetime(df_forecast['date'], format='%Y-%m-%d').dt.date

today = date.today()
yesterday = date.today() + timedelta(days=-1) 
tomorrow = date.today() + timedelta(days=1)

last_date_prev_month = today + timedelta(days=-(today.day))
first_date_prev_month = last_date_prev_month.replace(day=1)

init_date = datetime.strptime("2024-05-01", '%Y-%m-%d')

lag_range = 5
data_lag = 250
days_deep = 14

def pycaret_model(df, forecast_date, tr='open'):

	global data_lag, lag_range

	data = df[['date', tr]].copy()
	data.set_index('date', inplace=True)

	for lag in range(1, lag_range+1):
		data[f'Lag_{lag}'] = data[tr].shift(lag)

	data_lagged = data.dropna()
	data_lagged = data_lagged.tail(data_lag)

	data_lagged.sort_index(ascending=True, inplace=True)
	
	historic_data = data_lagged[data_lagged.index < forecast_date].copy()

	s = setup(data = historic_data, target = tr, train_size = 0.8, session_id=123)

	best = compare_models()
	model = create_model(best)

	NewData = data_lagged.tail(1)

	record = {}

	for lag in range(1, lag_range+1):
		record[f'Lag_{lag}'] = data_lagged[tr].tail(lag)[0]

	NewData = pd.DataFrame(record, index=[forecast_date])   # datetime.strptime(   , '%Y-%m-%d')
	NewData.index.name='date'

	forecasts=predict_model(model, data=NewData)


	return forecasts

def model_choice():

	global df_history, df_forecast, today, yesterday,  tomorrow, last_date_prev_month,  first_date_prev_month, init_date, days_deep


	tab1, tab2, tab3 = st.tabs(["Today", "Tomorrow", "Historical"])

	tab1.subheader(today.strftime('%Y-%m-%d') + ' Forecast')

	with tab1:

		open_forecast = df_forecast[df_forecast['date']==today]['open'].tolist()
		close_forecast = df_forecast[df_forecast['date']==today]['close'].tolist()

		if (len(open_forecast)>0):
			st.write(f"Open	 :   {open_forecast[0]:10.6f}")
			st.write(f"Close	:   {close_forecast[0]:10.6f}")
		else:

#			forecast_open = pycaret_model(df_history, today, 'open')
#			forecast_close = pycaret_model(df_history, today, 'close')

			open_forecast = "N/A" #forecast_open['prediction_label'][-1]
			close_forecast = "N/A" #forecast_close['prediction_label'][-1]

#			st.write(f"Open	 :   {open_forecast:10.6f}")
#			st.write(f"Close	:   {close_forecast:10.6f}")


			st.write(f"Open	 :   {open_forecast}")
			st.write(f"Close	:   {close_forecast}")



		y_min = df_history[df_history['date']>(today+timedelta(days=-days_deep))]['open'].min()  # Adjust the y-axis minimum value
		y_max = df_history[df_history['date']>(today+timedelta(days=-days_deep))]['open'].max()  # Adjust the y-axis maximum value
		y_padding = (y_max - y_min) * 0.1  # Add some padding to the y-axis range

		y_domain = [y_min - y_padding, y_max + y_padding]  # Adjusted y-axis domain range

		
#		st.line_chart(df_history[df_history['date']>(today+timedelta(days=-days_deep))], x="date", y=['open', 'close'], color=["#ff0000", "#0099"])

		data = df_history[df_history['date']>(today+timedelta(days=-days_deep))][['date', 'open', 'close']]

		st.write('Historical evalution')

		line_chart = alt.Chart(data).mark_line().encode(
    							x=alt.X('date:T', axis=alt.Axis(title='Date', grid=True, 
                                  format='%Y-%m-%d', labelAngle=-45)
                                  ),

							    y=alt.Y(alt.repeat('layer'),
						            scale=alt.Scale(domain=y_domain)
            						).aggregate('mean').title("Open and Close Prices"), color=alt.ColorDatum(alt.repeat('layer'))
								).repeat(layer=["open", "close"])

		st.altair_chart(line_chart, use_container_width=True)


	tab2.subheader(tomorrow.strftime('%Y-%m-%d') + ' Forecast')

	with tab2:

		open_forecast = df_forecast[df_forecast['date']==tomorrow]['open'].tolist()
		close_forecast = df_forecast[df_forecast['date']==tomorrow]['close'].tolist()	


		if (len(open_forecast)):	
			st.write(f"Open	 :   {open_forecast[0]:10.6f}")
			st.write(f"Close	:   {close_forecast[0]:10.6f}")

		else:

#			forecast_open = pycaret_model(df_history, tomorrow, 'open')
#			forecast_close = pycaret_model(df_history, tomorrow, 'close')

#			open_forecast = forecast_open['prediction_label'][-1]
#			close_forecast = forecast_close['prediction_label'][-1]

#			st.write(f"Open	 :   {open_forecast:10.6f}")
#			st.write(f"Close	:   {close_forecast:10.6f}")

			open_forecast = "N/A" #forecast_open['prediction_label'][-1]
			close_forecast = "N/A" #forecast_close['prediction_label'][-1]

			st.write(f"Open	 :   {open_forecast}")
			st.write(f"Close	:   {close_forecast}")



#		st.line_chart(df_history[df_history['date']>(today+timedelta(days=-days_deep))], x="date", y=['open', 'close'], color=["#ff0000", "#0099"])

		y_min = df_history[df_history['date']>(today+timedelta(days=-days_deep))]['open'].min()  # Adjust the y-axis minimum value
		y_max = df_history[df_history['date']>(today+timedelta(days=-days_deep))]['open'].max()  # Adjust the y-axis maximum value
		y_padding = (y_max - y_min) * 0.1  # Add some padding to the y-axis range

		y_domain = [y_min - y_padding, y_max + y_padding]  # Adjusted y-axis domain range


		data = df_history[df_history['date']>(today+timedelta(days=-days_deep))][['date', 'open', 'close']]

		st.write('Historical evalution')


		line_chart = alt.Chart(data).mark_line().encode(
    							x=alt.X('date:T', axis=alt.Axis(title='Date', grid=True, 
                                  format='%Y-%m-%d', labelAngle=-45)
                                  ),

							    y=alt.Y(alt.repeat('layer'),
						            scale=alt.Scale(domain=y_domain)
            						).aggregate('mean').title("Open and Close Prices"), color=alt.ColorDatum(alt.repeat('layer'))
								).repeat(layer=["open", "close"])

		st.altair_chart(line_chart, use_container_width=True)



	tab3.subheader("Forecast vs Fact")
	
	with tab3:

		df_test = pd.merge(df_forecast, df_history, on='date', how='left', suffixes=('_forecast', '_history'))
		df_test['open_pct'] = (df_test['open_forecast'] - df_test['open_history'])/df_test['open_history']*100
		df_test['close_pct'] = (df_test['close_forecast'] - df_test['close_history'])/df_test['close_history']*100

		df_test = df_test[['date', 'open_forecast', 'open_history', 'close_forecast', 'close_history', 'open_pct', 'close_pct']]

		selected_dates = st.date_input('Select period', value=(yesterday+timedelta(days=-days_deep*2),yesterday), min_value=init_date, max_value=yesterday, format="YYYY/MM/DD")

		start_date = selected_dates[0]
		end_date = selected_dates[1]

		df_test = df_test[(df_test['date']>=start_date) & (df_test['date']<=end_date)]
		df_test.dropna(subset=['open_history'], inplace=True)


		if st.button("Send", key="button1"): 

			st.dataframe(data=df_test[['date', 'open_forecast', 'open_history','open_pct', 'close_forecast', 'close_history', 'close_pct']], hide_index=True, column_config=column_config_h)

			st.write('Open')

			y_min = df_test[df_test['date']>(today+timedelta(days=-days_deep*2))]['open_history'].min()  # Adjust the y-axis minimum value
			y_max = df_test[df_test['date']>(today+timedelta(days=-days_deep*2))]['open_history'].max()  # Adjust the y-axis maximum value
			y_padding = (y_max - y_min) * 0.1  # Add some padding to the y-axis range

			y_domain = [y_min - y_padding, y_max + y_padding]  # Adjusted y-axis domain range


			line_chart = alt.Chart(df_test[['date','open_forecast', 'open_history']]).mark_line().encode(
    							x=alt.X('date:T', axis=alt.Axis(title='Date', grid=True, 
                                  format='%Y-%m-%d', labelAngle=-45)
                                  ),

							    y=alt.Y(alt.repeat('layer'),
						            scale=alt.Scale(domain=y_domain)
            						).aggregate('mean').title("Open Prices"), color=alt.ColorDatum(alt.repeat('layer'))
								).repeat(layer=['open_forecast', 'open_history'])

			st.altair_chart(line_chart, use_container_width=True)

			st.write('Close')

			y_min = df_test[df_test['date']>(today+timedelta(days=-days_deep*2))]['close_history'].min()  # Adjust the y-axis minimum value
			y_max = df_test[df_test['date']>(today+timedelta(days=-days_deep*2))]['close_history'].max()  # Adjust the y-axis maximum value
			y_padding = (y_max - y_min) * 0.1  # Add some padding to the y-axis range

			y_domain = [y_min - y_padding, y_max + y_padding]  # Adjusted y-axis domain range


			line_chart = alt.Chart(df_test[['date','close_forecast', 'close_history']]).mark_line().encode(
    							x=alt.X('date:T', axis=alt.Axis(title='Date', grid=True, 
                                  format='%Y-%m-%d', labelAngle=-45)
                                  ),

							    y=alt.Y(alt.repeat('layer'),
						            scale=alt.Scale(domain=y_domain)
            						).aggregate('mean').title("Close Prices"), color=alt.ColorDatum(alt.repeat('layer'))
								).repeat(layer=['close_forecast', 'close_history'])



			st.altair_chart(line_chart, use_container_width=True)

			st.write('Error')
			st.line_chart(df_test, x="date", y=['open_pct', 'close_pct'], color=["#ff0000", "#0099"])



def main():
	
	output_type=None
	output = model_choice()


if __name__ == "__main__":
	main()