# -*- coding: utf-8 -*-
"""
Account_Usage_OLS_Model

Created on Wed Dec 20 11:45:17 2023

@author: rvackner
"""

# import libraries
import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.api as sm
import statsmodels.tsa.tsatools as tsa

# import account, usage, and weather data
cnxncis = pyodbc.connect('DSN=XXXXX;PWD=XXXXX')
cnxnsql = pyodbc.connect('DRIVER=XXXXX;SERVER=XXXXX;DATABASE=XXXXX;Trusted_Connection=XXXXX')
df_mdm_daily = pd.read_sql_query("SELECT * FROM XXXXX.XXXXX", cnxnsql)
df_weather_data = pd.read_sql_query("SELECT * FROM XXXXX.XXXXX WHERE date >= CONVERT(DATETIME, '1/1/2022')", cnxnsql)
df_type_service = pd.read_sql_query("SELECT BI_ACCT FROM XXXXX.XXXXX WHERE (BI_SRV_STAT_CD = 1) or (BI_SRV_STAT_CD = 18)", cnxncis)
df_rdg = pd.read_sql_query("SELECT BI_ACCT, BI_MTR_NBR, BI_MTR_MULT FROM XXXXX.XXXXX", cnxncis)

# fix all data types
df_mdm_daily["Recorded_Date"] = pd.to_datetime(df_mdm_daily["Recorded_Date"]).dt.date
df_mdm_daily['Meter_Number'] = pd.to_numeric(df_mdm_daily['Meter_Number'], errors='coerce')
df_mdm_daily['Cumulative_Usage'] = pd.to_numeric(df_mdm_daily['Cumulative_Usage'], errors='coerce')
df_mdm_daily['In_Window_Demand'] = pd.to_numeric(df_mdm_daily['In_Window_Demand'], errors='coerce')
df_mdm_daily['All_Day_Demand'] = pd.to_numeric(df_mdm_daily['All_Day_Demand'], errors='coerce')
df_weather_data["date"] = pd.to_datetime(df_weather_data["date"]).dt.date
df_rdg['BI_MTR_NBR'] = pd.to_numeric(df_rdg['BI_MTR_NBR'], errors='coerce')
df_rdg['BI_ACCT'] = pd.to_numeric(df_rdg['BI_ACCT'], errors='coerce')

# get only active accounts and current meter nbr
df_rdg = df_rdg[df_rdg['BI_ACCT'].isin(list(df_type_service['BI_ACCT']))]
df_rdg = df_rdg.groupby(['BI_ACCT', 'BI_MTR_NBR'])['BI_MTR_MULT'].max().reset_index()

# add acct number by merging on meter number
df_daily_usage = df_mdm_daily.merge(df_rdg, how='inner', left_on="Meter_Number", right_on="BI_MTR_NBR")
# add weather by merging on date
df_daily_usage = df_daily_usage.merge(df_weather_data, how='left', left_on='Recorded_Date', right_on='date')
# remove dates where we don't have weather data
df_daily_usage = df_daily_usage[~df_daily_usage["temp_avg"].isnull()]
# use mtr mult to get actual usage and demand
df_daily_usage['Cumulative_Usage'] = df_daily_usage['BI_MTR_MULT']*df_daily_usage['Cumulative_Usage']
df_daily_usage['In_Window_Demand'] = df_daily_usage['BI_MTR_MULT']*df_daily_usage['In_Window_Demand']
df_daily_usage['All_Day_Demand'] = df_daily_usage['BI_MTR_MULT']*df_daily_usage['All_Day_Demand']
# add a column for the day of the week
df_daily_usage['Day'] = (pd.to_datetime(df_daily_usage['Recorded_Date'])).dt.day_name()
# encode dummies hot key for day of the week
df_daily_usage[['fri', 'mon', 'sat', 'sun', 'thur', 'tues', 'wed']] = pd.get_dummies(df_daily_usage['Day'])
# add holidays
holidays = list(USFederalHolidayCalendar().holidays(start='2022-01-01', end=datetime.today().strftime('%Y-%m-%d')).date)
df_daily_usage['holiday'] = np.where(df_daily_usage['Recorded_Date'].isin(holidays), 1, 0)


# daily usage regression function
def use_reg_func(df):
    # account must have atleast a full years worth of data to be accurate
    if len(df.index) >= 365:
        # get the account to label the pickled file
        acct = int(df['BI_ACCT'].max())
        # add the squared degree days values
        df[['hdd', 'cdd']] = df[['hdd', 'cdd']].astype(float)
        df['hdd2'] = df['hdd']*df['hdd']
        df['cdd2'] = df['cdd']*df['cdd']
        # add a constant trend line
        x = tsa.add_trend(df[['hdd', 'cdd', 'hdd2', 'cdd2', 'sun', 'mon', 'tues', 'wed', 'thur', 'fri', 'sat', 'holiday']], 'c')
        # fit the model, this is what we will save and use to make predictions
        fitted_model = sm.OLS(df['Cumulative_Usage'], x).fit()
        fitted_model.save(fr"XXXXX\{acct}.pickle")

# daily demand regression function
def dmd_reg_func(df):
    # account must have atleast a full years worth of data to be accurate
    if len(df.index) >= 365:
        # get the account to label the pickled file
        acct = int(df['BI_ACCT'].max())
        # add the squared degree days values
        df[['hdd', 'cdd']] = df[['hdd', 'cdd']].astype(float)
        df['hdd2'] = df['hdd']*df['hdd']
        df['cdd2'] = df['cdd']*df['cdd']
        # add a constant trend line
        x = tsa.add_trend(df[['hdd', 'cdd', 'hdd2', 'cdd2', 'sun', 'mon', 'tues', 'wed', 'thur', 'fri', 'sat', 'holiday']], 'c')
        # fit the model, this is what we will save and use to make predictions
        fitted_model = sm.OLS(df['In_Window_Demand'], x).fit()
        fitted_model.save(fr"XXXXX\{acct}.pickle")

# call both use and dmd regression functions
df_daily_usage.groupby('BI_ACCT').apply(use_reg_func)
df_daily_usage.groupby('BI_ACCT').apply(dmd_reg_func)
