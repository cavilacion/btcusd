from datetime import date,timedelta,datetime
from time import mktime
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import os

class CustomDataFrame:
    def __init__(self,fr,to):
        self.api_key=os.environ["POLYGON_API"]
        self.fr = fr
        self.to = to

    def request(self):
        rest_url = f"https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/range/1/hour/{self.fr}/{self.to}?adjusted=true&sort=asc&apiKey={self.api_key}"
        response = requests.get(rest_url)
        json_str = response.json()
        try:
            self.data=pd.DataFrame.from_dict(json_str['results'])
        except KeyError:
            print("\nFailed to get data. Skipping.")
            return None,0

    def query(self,unixtime):
        selection = self.data.loc[self.data["t"]==unixtime]
        return selection.rename(columns={"o":"Open","c":"Close","h":"High","l":"Low","v":"Volume","t":"Datetime"})

df = pd.read_csv("data/raw_data.csv")
df['Datetime'] = pd.to_datetime(df["Datetime"])
timestamps = df["Datetime"]
time_diffs = (df["Datetime"]-df["Datetime"].shift(1)).astype('timedelta64[ns]')
for i in range(1,len(time_diffs)):
    h = time_diffs[i]/np.timedelta64(1,'h')
    if h != 1:
        print (f"healing from {timestamps[i-1]} to {timestamps[i]}")

        fr = timestamps[i-1].date().strftime("%Y-%m-%d")
        to = timestamps[i].date().strftime("%Y-%m-%d")
        data = CustomDataFrame(fr,to)
        data.request()
        
        curr_ts = timestamps[i-1]+timedelta(hours=1)
        while curr_ts < timestamps[i]:
            unixtime = int(mktime(curr_ts.timetuple())*1000)
            new_row = data.query(unixtime)
            new_row["Datetime"]=curr_ts
            new_row = new_row[["Datetime","Close","Open","High","Low","Volume"]]
            df = pd.concat([df,new_row])
            curr_ts += timedelta(hours=1)


df = df.sort_values(by='Datetime')
df.to_csv("data/healed_data.csv",index=False)

df = pd.read_csv("data/healed_data.csv")
df['Datetime'] = pd.to_datetime(df["Datetime"])
timestamps = df["Datetime"]
time_diffs = (df["Datetime"]-df["Datetime"].shift(1)).astype('timedelta64[ns]')
for i in range(1,len(time_diffs)):
    h = time_diffs[i]/np.timedelta64(1,'h')
    if h != 1:
        print (f"healing data failed!")

