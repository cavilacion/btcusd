import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date,timedelta,datetime
import numpy as np

N=730
T  = datetime.now()
t0 = T-timedelta(days=N) 

data=yf.download("BTC-USD", interval="1h", progress=False, start=t0, end=T,)
plt.plot(data['Open'])
model_data=pd.DataFrame(data)
model_data.to_csv(f"data/raw_data.csv")
plt.show()

