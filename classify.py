from datetime import date,timedelta,datetime
from time import mktime
import pandas as pd
import requests
import json
import numpy as np
from math import log

map_decs = {"incr":0, "decr":1, "noact":2}

class Classification:
    def __init__(self, symbol, category="tech", step=0.1, num_bins=10, filepath=None, verbose=True):
        self.symbol=symbol 
        self.category=category
        self.step = step
        self.num_bins = num_bins
        if filepath is None:
            filepath = "/".join(["data",category,symbol+".csv"])
        self.verbose=verbose
        self.df = pd.read_csv(filepath,index_col=0)
        self.X = self.df[["close"]].to_numpy().flatten() 
        self.hist = np.array([0]*num_bins)
        self.bounds = [None]*num_bins

    def make_hist (self):
        N = len(self.X)
        diffs = [None]*N
        max_diff = -1
        for i in range(1,N):
            diffs[i] = abs(self.X[i]-self.X[i-1])
            if diffs[i] > max_diff:
                max_diff = diffs[i]
    
        for i in range(1,self.num_bins+1):
            self.bounds[i-1] = 1.0*i*max_diff/self.num_bins
    
        for i in range(1,N):
            for j in range(self.num_bins):
                if diffs[i] <= self.bounds[j]:
                    self.hist[j] += 1
                    break

    def max_threshold (self):
        temp_sum = 0
        sum_bin_counts = self.hist.sum()
        for i in range(self.num_bins):
            temp_sum = temp_sum + self.hist[i]
            if temp_sum / sum_bin_counts >= 0.85:
                break 
        return self.bounds[i]

    def get_class_probs (self,thr):
        N = len(self.X)
        counts = {"incr":0, "decr":0, "noact":0}
        for i in range(1,N):
            diff = self.X[i]-self.X[i-1]
            if abs(diff) <= thr:
                counts["noact"] += 1
            elif diff > 0:
                counts["incr"] += 1
            else:
                counts["decr"] += 1
        p_incr  = 1.0*counts["incr"]/(N-1)
        p_decr  = 1.0*counts["decr"]/(N-1)
        p_noact = 1.0*counts["noact"]/(N-1)
        return {"incr":p_incr,"decr":p_decr,"noact":p_noact}
    
    def entropy (self, ps):
        return -( ps["incr"]*log(ps["incr"],3)\
                 +ps["decr"]*log(ps["decr"],3)\
                 +ps["noact"]*(0 if ps["noact"]==0 else log(ps["noact"],3)))

    def set_threshold (self):
        self.make_hist()
        max_threshold = self.max_threshold()
        
        max_entropy = -9999999999
        running_threshold = 0

        N=0
        total = int(max_threshold/self.step)

        while running_threshold <= max_threshold:
            probs = self.get_class_probs(running_threshold)
            ent = self.entropy (probs)
            if ent > max_entropy:
                max_entropy = ent 
                threshold = running_threshold
            running_threshold += self.step
            N+=1

            if self.verbose and N%100==0:
                print(f"{N}/{total}")

        self.threshold = threshold 
        self.max_entropy = max_entropy
        if self.verbose:
            print(f"Using threshold {threshold} with entropy {max_entropy}")

    def classify_returns (self):
        self.set_threshold()
        returns = [None]*len(self.X)
        for i in range(1,len(self.X)):
            returns[i] = (self.X[i]-self.X[i-1])
        self.returns = returns
        self.df.insert(len(self.df.columns),"decision",returns)
    
    def write_csv(self,filepath=None):
        if filepath is None:
            filepath = "/".join(["data",self.category,f"{self.symbol}_labeled.csv"])
        self.df.to_csv(filepath)

cl = Classification ("eurusd",filepath="data/tech_data.csv")
cl.classify_returns()
cl.write_csv(filepath="data/tech_data.csv")
