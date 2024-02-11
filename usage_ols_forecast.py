# -*- coding: utf-8 -*-
"""
Account_Usage_OLS_Model_Prediction

Created on Thu Dec 21 16:25:07 2023

@author: rvackner
"""

# import libraries
import os
import pyodbc
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pandas.tseries.holiday import USFederalHolidayCalendar
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import weather_data as wd


def get_weather(location):
    weather_data = None
    while weather_data is None:
        try:
            weather_data = ((wd.WeatherGetter(location)).GetNwsWeather())
        except:
            weather_data = None
    n = 0
    avg_temp = []
    while n <= 12:
        temp = (weather_data['properties']['periods'][n]['temperature']+weather_data['properties']['periods'][n+1]['temperature'])/2
        avg_temp.append(temp)
        n+=2
    return avg_temp

cnxnsql = pyodbc.connect('DRIVER=XXXXX;SERVER=XXXXX;DATABASE=XXXXX;Trusted_Connection=XXXXX')

df_weather_data = pd.read_sql_query("SELECT * FROM XXXXX.XXXXX WHERE date < CONVERT(date, getdate()) AND date > CONVERT(date, getdate()-8)", cnxnsql)
weather_forecast = list(df_weather_data['temp_avg'].astype(float))

df_locations = pd.DataFrame([['Cades', 33.8083, -79.8566, np.nan, np.nan, np.nan]], columns=['name', 'lat', 'long', 'temp_td', 'temp_tn', 'temp_avg'])

forecast = sum(list(df_locations.apply(lambda x: get_weather(x.to_dict()), axis=1)), [])
weather_forecast = weather_forecast + forecast
datelist = pd.date_range(date.today()-relativedelta(days=7), end=date.today()+relativedelta(days=6)).tolist()
holidays = list(USFederalHolidayCalendar().holidays(start=datelist[0], end=datelist[-1]).date)

dd_base = 65
df = pd.DataFrame()
df['Recorded_Date'] = datelist
df['Avg_Temp'] = weather_forecast
df['hdd'] = np.where(df["Avg_Temp"] > dd_base, 0, dd_base-df["Avg_Temp"])
df['cdd'] = np.where(df["Avg_Temp"] < dd_base, 0, df["Avg_Temp"]-dd_base)
df['hdd2'] = df['hdd']*df['hdd']
df['cdd2'] = df['cdd']*df['cdd']
df['Day'] = (pd.to_datetime(df['Recorded_Date'])).dt.day_name()
df[['fri', 'mon', 'sat', 'sun', 'thur', 'tues', 'wed']] = pd.get_dummies(df['Day'])
df['holiday'] = np.where(df['Recorded_Date'].isin(holidays), 1, 0)
df['const'] = 1.0
df_weather_data = df[['Recorded_Date', 'Avg_Temp']]
df = df[['hdd', 'cdd', 'hdd2', 'cdd2', 'sun', 'mon', 'tues', 'wed', 'thur', 'fri', 'sat', 'holiday', 'const']]


df_prediction = pd.DataFrame(columns=['BI_ACCT', 'Recorded_Date', 'use_prediction', 'use_r_squared', 'use_rse', 'dmd_prediction', 'dmd_r_squared', 'dmd_rse'])
use_directory = r'XXXXX\pickled_use_reg'
dmd_directory = r'XXXXX\pickled_dmd_reg'


for filename in os.listdir(use_directory):
    f_use = os.path.join(use_directory, filename)
    reg_use = sm.load(f_use)
   
    f_dmd = os.path.join(dmd_directory, filename)
    reg_dmd = sm.load(f_dmd)
   
    for i, row in df.iterrows():
        acct = int(os.path.splitext(filename)[0])
        date = datelist[i]
       
        prediction_use = getattr(reg_use.predict(row), 'array')[0]
        r_squared_use = reg_use.rsquared
        rse_use = np.sqrt(reg_use.scale)
       
        prediction_dmd = getattr(reg_dmd.predict(row), 'array')[0]
        r_squared_dmd = reg_dmd.rsquared
        rse_dmd = np.sqrt(reg_dmd.scale)
       
        df_prediction.loc[len(df_prediction.index)] = [acct, date, prediction_use, r_squared_use, rse_use, prediction_dmd, r_squared_dmd, rse_dmd]


df_prediction = df_prediction.merge(df_weather_data, how='left', on='Recorded_Date')

cnxncis = pyodbc.connect('DSN=XXXXX;PWD=XXXXX')
df_rdg = pd.read_sql_query("SELECT BI_ACCT, BI_MTR_NBR, BI_MTR_MULT, BI_RATE_SCHED FROM XXXXX.XXXXX", cnxncis)
df_rdg['BI_ACCT'] = pd.to_numeric(df_rdg['BI_ACCT'], errors='coerce')
df_rdg['BI_MTR_NBR'] = pd.to_numeric(df_rdg['BI_MTR_NBR'], errors='coerce')
df_mdm = pd.read_sql_query(f"SELECT Meter_Number AS BI_MTR_NBR, Recorded_Date, Cumulative_Usage, In_Window_Demand FROM XXXXX.XXXXX WHERE CAST(Recorded_Date AS DATE)>='{str(datelist[0].date())}'", cnxnsql)
df_mdm['BI_MTR_NBR'] = df_mdm['BI_MTR_NBR'].astype(int)
df_mdm['Recorded_Date'] = pd.to_datetime(df_mdm['Recorded_Date'])
df_mdm['Cumulative_Usage'] = df_mdm['Cumulative_Usage'].astype(float)
df_mdm['In_Window_Demand'] = df_mdm['In_Window_Demand'].astype(float)

df_actual = df_mdm.merge(df_rdg, how='left', on='BI_MTR_NBR')
df_actual['Cumulative_Usage'] = df_actual['Cumulative_Usage']*df_actual['BI_MTR_MULT']
df_actual['In_Window_Demand'] = df_actual['In_Window_Demand']*df_actual['BI_MTR_MULT']

df_plot = df_prediction.merge(df_actual, how='left', on=['BI_ACCT', 'Recorded_Date'])
df_plot['BI_RATE_SCHED'] = df_plot['BI_RATE_SCHED'].fillna(method='ffill')
df_plot['BI_MTR_NBR'] = df_plot['BI_MTR_NBR'].fillna(method='ffill')
df_plot['use_upper_lim'] = df_plot['use_prediction']+(3*df_plot['use_rse'])
df_plot['use_lower_lim'] = df_plot['use_prediction']-(3*df_plot['use_rse'])
df_plot['dmd_upper_lim'] = df_plot['dmd_prediction']+(3*df_plot['dmd_rse'])
df_plot['dmd_lower_lim'] = df_plot['dmd_prediction']-(3*df_plot['dmd_rse'])

df_plot['use_difference'] = df_plot['Cumulative_Usage'] - df_plot['use_prediction']
df_plot['use_percent_change'] = (df_plot['use_difference']/df_plot['use_prediction'])*100
df_plot['use_z_score'] = df_plot['use_difference']/df_plot['use_rse']
df_plot['dmd_difference'] = df_plot['In_Window_Demand'] - df_plot['dmd_prediction']
df_plot['dmd_percent_change'] = (df_plot['dmd_difference']/df_plot['dmd_prediction'])*100
df_plot['dmd_z_score'] = df_plot['dmd_difference']/df_plot['dmd_rse']

df_plot.to_csv(r"XXXXX\use_dmd_pred.csv", index=False)
