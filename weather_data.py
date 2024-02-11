"""
Get Weather Data

Created on Fri Jun 16 15:54:03 2023

@author: rvackner
"""

import requests
import json
import pyodbc
import pandas as pd
import time
import numpy as np
import datetime
from datetime import date

class WeatherGetter:
    def __init__(self, location):
        self.location = location
         
         
    def __ApiCall(self, url, headers):
        response = requests.get(url, headers = headers)

        return json.loads(response.content.decode('utf-8'))
   
   
    def __GetNwsPoints(self):
        lat = self.location["lat"]
        lon = self.location["long"]

        url = f"https://api.weather.gov/points/{lat},{lon}"

        headers = {
            "User-Agent": "(XXXXX, XXXXX)"
        }

        return self.__ApiCall(url, headers)
   
   
    def GetNwsWeather(self):
        points = self.__GetNwsPoints()

        office = points["properties"]["gridId"]
        gridX = points["properties"]["gridX"]
        gridY = points["properties"]["gridY"]

        url = f"https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast"

        headers = {
            "User-Agent": "(XXXXX, XXXXX)"
        }

        return self.__ApiCall(url, headers)
   
   
   
def get_weather(location):
    try:
        weather_data = ((WeatherGetter(location)).GetNwsWeather())
        temp_td = weather_data['properties']['periods'][0]['temperature']
        temp_tn = weather_data['properties']['periods'][1]['temperature']
        temp_avg = int(round((temp_td+temp_tn)/2))
    except:
        temp_td = None
        temp_tn = None
        temp_avg = None
    return temp_td, temp_tn, temp_avg
