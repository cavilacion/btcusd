import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pickle
import torch
from torch import nn,Tensor
from torch.autograd import Variable
import keras
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import time

from metrics import CustomCategoricalEntropyLoss, MeanAbsoluteDirectionalLoss,\
    ProfitPerformance, CustomAccuracy, BetUp, map_decs,\
    profit_performance, transaction_ratio

class Model:
    def __init__(self, 
                 look_back=5,
                 num_iterations=1,
                 num_epochs=300,
                 learning_rate=2.15,
                 num_layers=1,
                 hidden_size=128,
                 dropout=0.02,
                 category="tech",
                 filepath=None):
        self.category = category
        self.look_back = look_back
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_layers=num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        if filepath is None:
            filepath = "/".join(["data",category,symbol+"_labeled.csv"])
        self.df = pd.read_csv(filepath,index_col=0)
        self.cols = ['close', 
                         'ma',
                         'macd',
                         'roc',
                         'rsi',
                         'mom',
                         'bol_d',
                         'bol_u',
                         'cci']

    def time_transform_data(self, X_in, y_in):
        X_out,y_out = [], []
        for i in range(len(X_in)-self.look_back):
            X_out.append(X_in[i:(i+self.look_back),:].flatten())
            y_out.append(y_in[i+self.look_back])
        return np.array(X_out), np.array(y_out)

    def prepare_data(self):
        X = self.df[self.cols].values.astype('float32')
        y = self.df[["return"]].values.flatten()

        # normalize X data
        self.num_features = len(self.cols)
        self.data_scaler = MinMaxScaler(feature_range=(0, 1))
        X = self.data_scaler.fit_transform(X)

        # incorporate look back
        X, y = self.time_transform_data(X, y)

        # split in train and test data
        while len(X)%5 != 0:
            X = X[1:]
            y = y[1:]
        self.train_size = int(0.8*X.shape[0])
        self.test_size = X.shape[0] - self.train_size
        X_train = X[:self.train_size,:]
        y_train = y[:self.train_size]
        X_test  = X[self.train_size:,:]
        y_test  = y[self.train_size:]
        self.batch_size = X_test.shape[0]

        # shape train data
        X_shape = (X_train.shape[0],self.look_back,self.num_features)
        X_train = np.reshape(X_train,X_shape)
        X_train = Variable(Tensor(X_train))
        y_train = Variable(Tensor(y_train))

        # shape test data
        X_shape = (X_test.shape[0],self.look_back,self.num_features)
        X_test = np.reshape(X_test,X_shape)
        X_test = Variable(Tensor(X_test))
        y_test = Variable(Tensor(y_test))
        
        self.X_train,self.y_train = X_train,y_train
        self.X_test,self.y_test   = X_test,y_test

        print("Training Shape",self.X_train.shape, self.y_train.shape)
        print("Testing Shape", self.X_test.shape, self.y_test.shape)

    def train(self):
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = keras.Sequential()
        self.model.add(keras.layers.InputLayer(batch_input_shape=(self.batch_size,self.look_back,self.num_features)))

        size = self.hidden_size
        for layer_no in range(self.num_layers-1):
            self.model.add(keras.layers.LSTM(size, stateful=True, dropout=self.dropout, return_sequences=True))
            size //= 2
        self.model.add(keras.layers.LSTM(size, stateful=True, dropout=self.dropout, return_sequences=False))
        self.model.add(keras.layers.Dense(1))
        self.model.compile(loss=MeanAbsoluteDirectionalLoss(),
                      optimizer='adam',
                      metrics=[CustomAccuracy(),BetUp()],
                      run_eagerly=True)
        return self.model.fit(self.X_train, self.y_train,
                  epochs=self.num_epochs,
                  batch_size=self.batch_size,
                  validation_data=(self.X_test,self.y_test),
                  verbose=2)
        self.model.summary()

    def get_test_results(self):
        self.new_model = keras.Sequential()

        self.new_model.add(keras.layers.InputLayer(batch_input_shape=(self.batch_size,self.look_back,self.num_features)))
        size = self.hidden_size
        for layer_no in range(self.num_layers-1):
            self.new_model.add(keras.layers.LSTM(size, stateful=True, dropout=self.dropout, return_sequences=True))
            size //= 2
        self.new_model.add(keras.layers.LSTM(size, stateful=True, dropout=self.dropout, return_sequences=False))
        self.new_model.add(keras.layers.Dense(1))

        model_weights = self.model.get_weights()
        self.new_model.set_weights(model_weights)
        self.new_model.compile(loss=CustomCategoricalEntropyLoss(),
                      optimizer='adam',
                      metrics=[ProfitPerformance(),CustomAccuracy()],
                      run_eagerly=False)
        return self.new_model.predict(self.X_test)

        if len(self.X_test) % self.batch_size != 0:
            print(f"Test size {len(self.X_test)} not appropriate"+\
                  f"for batch size {self.batch_size}")
        num_iter = int(len(self.X_test)/self.batch_size)
        test_pred = []
        test_pred = self.model.predict(self.X_test)
        #for i in range(num_iter):
            #a = i*self.batch_size 
            #b = (i+1)*self.batch_size
            #test_input = self.X_test[a:b]
            #test = self.model.predict(test_input)
            #test_pred.append(np.array(test))
        return np.concatenate(test_pred),np.array(self.y_test)

    def predict(self):
        self.new_model = keras.Sequential()
        self.new_model.add(keras.layers.InputLayer(batch_input_shape=(1,self.look_back,self.num_features)))
        self.new_model.add(keras.layers.LSTM(self.hidden_size, stateful=True, dropout=self.dropout))
        self.new_model.add(keras.layers.Dense(3))
        self.new_model.add(keras.layers.Activation('sigmoid'))
        model_weights = self.model.get_weights()
        self.new_model.set_weights(model_weights)
        self.new_model.compile(loss=CustomCategoricalEntropyLoss(),
                      optimizer='adam',
                      metrics=[ProfitPerformance(),TransactionRatio()],
                      run_eagerly=False)
        pred_y = self.new_model.predict(self.X_predict)
        print(pred_y)


def run_fund():
    for i in range(5):
        fm = Model(category="fund",
           num_epochs=50,
           learning_rate=1e-12,
           num_iterations=200,
           hidden_size=8,
           dropout=0.1,
           filepath="data/eurusd_all_labeled.csv")
        fm.prepare_data()
        history = fm.train()
        plt.plot(history.history['loss'], color='blue')
        plt.plot(history.history['val_loss'], color='orange')
        plt.plot(history.history['profit_performance'], color='blue',linestyle='--')
        plt.plot(history.history['val_profit_performance'], color='orange',linestyle='--')
    plt.grid()
    plt.legend(["train loss","test loss","train performance","test performance"])
    plt.show()
    y_pred,y_true = fm.get_test_results()
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    num,denom = transaction_ratio(y_pred)
    profitability = profit_performance(y_true,y_pred)
    print(f"transaction ratio: {num}/{denom}")
    print(f"profitability performance: {profitability}")

def run_tech():
    colors = ['blue','red','green','purple','orange','pink','brown']
    N = 1
    for t in range(5):
        for i in range(N):
            keras.utils.set_random_seed(int(time.time()))
            fm = Model(category="tech",
                   num_epochs=10,
                   learning_rate=2.15,
                   num_layers=3,
                   hidden_size=256,
                   dropout=0.08,
                   look_back=2,
                   filepath="data/tech_data.csv")
            fm.prepare_data()
            history = fm.train()
            plt.plot(history.history['loss'], color=colors[i])
            plt.plot(history.history['val_loss'], color=colors[i], linestyle='--')
        #plt.plot(history.history['profit_performance'], color='blue',linestyle='--')
        #plt.plot(history.history['val_profit_performance'], color='orange',linestyle='--')
    plt.grid()
    N = 2
    s = [f"h = {12+i*108}" for i in range(N)]
    s_train = [t+" (train)" for t in s]
    s_test = [t+" (test)" for t in s]
    plt.legend(np.array([[s_train[i],s_test[i]] for i in range(N)]).flatten())
    plt.show()
    y_pred,y_true = fm.get_test_results()
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    num,denom = transaction_ratio(y_pred)
    profitability = profit_performance(y_true,y_pred)
    print(f"transaction ratio: {num}/{denom}")
    print(f"profitability performance: {profitability}")

run_tech()
