from datetime import date,timedelta,datetime,UTC
from time import mktime
import pandas as pd
import requests
import json
import numpy as np
import statistics

class TechnicalIndicators:
    def __init__(self, symbol, filepath=None):
        self.symbol = symbol
        if filepath is None:
            filepath = f"data/{symbol}.csv"
        df = pd.read_csv(filepath)
        self.closes = df.loc[:,'Close'].to_numpy().flatten()
        self.opens = df.loc[:,'Open'].to_numpy().flatten()
        self.highs = df.loc[:,'High'].to_numpy().flatten()
        self.lows = df.loc[:,'Low'].to_numpy().flatten()
        self.volumes = df.loc[:,'Volume'].to_numpy().flatten()
        self.typicals = (self.lows + self.highs + self.closes) / 3
        self.timestamps = df.loc[:,'Datetime']

    def SMA (self, X=None, days=10):
        if X is None:
            X = self.closes
        if days < 1:
            print(f"Error: days must be at least 1")
            return []
        sma = [None]*len(X)
        S = X[:days].sum()
        sma = [None]*len(X)
        S = X[:days].sum()
        first = 0
        for last in range(days,len(X)):
            sma[last] = S / days
            S -= X[first]
            S += X[last]
            first += 1
            last += 1
        return sma 
    
    def EMA (self, days=10):
        X = self.closes
        ema = [None]*len(X)
        for i in range(len(X)):
            prev = X[0] if i<1 else ema[i-1]
            ema[i] = (X[i]*2/(1+days)) + prev*(1-2/(1+days))
        return ema
    
    def ROC (self, days=9):
        X = self.closes
        roc = [None]*len(X)
        for i in range(days,len(X)):
            roc[i] = (X[i]-X[i-days]) / X[i-days]
        return roc
    
    def Momentum (self, days=14):
        X = self.closes
        mom = [None]*len(X)
        for i in range(days,len(X)):
            mom[i] = X[i]-X[i-days]
        return mom
    
    def RSI (self, days=14):
        os = self.opens 
        cs = self.closes
        zero_floor = np.vectorize(lambda n: max(0,n),otypes=[float])
        Us = zero_floor(cs-os)
        Ds = zero_floor(os-cs)
        u = Us[:days].sum()
        d = Ds[:days].sum()
        rsi = [None]*len(os)
        first = 0
        for last in range(days,len(os)):
            avg_u = u/days 
            avg_d = d/days
            rs = avg_u/avg_d
            rsi[last] = 100-(100/(1+rs))
    
            u += (Us[last] - Us[first])
            d += (Ds[last] - Ds[first])
            first += 1
    
        return rsi
    
    def BOL (self, days=20, m=2):
        ts = self.typicals
        ma = self.SMA(ts,days)
        bolu = [None]*len(ts)
        bold = [None]*len(ts)
        for last in range(days,len(ts)):
            period = ts[last-days:last]
            sigma = statistics.stdev(period)
            bolu[last] = ma[last] + m*sigma 
            bold[last] = ma[last] - m*sigma
        return bold,bolu
    
    def CCI (self, days=20):
        ts = self.typicals
        ma = self.SMA(ts,days)
        cci = [None]*len(ts)
        for last in range(days,len(ts)):
            period = ts[(last-days):last]
            mean = statistics.mean(period)
            absolute = np.vectorize(lambda n: abs(n),otypes=[float])
            meandev = statistics.mean(absolute(period-mean))
            cci[last] = (ts[last]-ma[last]) / (0.015*meandev)
        return cci

    def ret (self):
        X = self.closes
        ret = [None]*len(X)
        for i in range(1,len(X)):
            ret[i] = X[i]-X[i-1]
        return ret
    


    def compute_indicators(self):
        self.MA10   = self.SMA()
        self.MACD   = np.array(self.EMA(12))\
                     -np.array(self.EMA(26))
        self.ROC9   = self.ROC()
        self.RSI14  = self.RSI()
        self.Mom14  = self.Momentum()
        bd,bu = self.BOL()
        self.BOLD20 = bd
        self.BOLU20 = bu
        self.CCI20  = self.CCI()

        self.rets   = self.ret()


    def write_csv(self,filepath=None):
        if filepath is None:
            filepath = f"data/tech/{self.symbol}.csv"
        df = pd.DataFrame (data={"timestamp": self.timestamps[20:],
                                 "close" : self.closes[20:],
                                 "open" : self.opens[20:],
                                 "high" : self.highs[20:],
                                 "low" : self.lows[20:],
                                 "volume" : self.volumes[20:],
                                 "ma" : self.MA10[20:],
                                 "macd": self.MACD[20:],
                                 "roc": self.ROC9[20:],
                                 "rsi": self.RSI14[20:],
                                 "mom": self.Mom14[20:],
                                 "bol_d": self.BOLD20[20:],
                                 "bol_u": self.BOLU20[20:],
                                 "cci": self.CCI20[20:],
                                 "return": self.rets[20:]})
        df.to_csv(filepath,index=False)

symbol = "btc-usd"
ti = TechnicalIndicators(symbol,filepath="data/healed_data.csv")
ti.compute_indicators()
ti.write_csv(filepath="data/tech_data.csv")

