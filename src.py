#导入相关包
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from collections import deque
import matplotlib.pyplot as plt
#导入数据并对数据预处理
Data = pd.read_csv('399300.csv',parse_dates=[0],index_col=[0])
Data['Adj Close'] = Data['Close']
Data['Adj_High'] = Data['High'] * Data['Adj Close'] / Data['Close']
Data['Adj_Low'] = Data['Low'] * Data['Adj Close'] / Data['Close']
Data['Adj_Open'] = Data['Open'] * Data['Adj Close'] / Data['Close']
pre_days = -10
Data = Data.drop(['Open', 'High', 'Low', 'Close','Volume'], axis=1)
dpIndex = []
dpIndex.extend(Data.index[-11:-1])
dpIndex.append(Data.index[-1])
Data = Data.drop(index = dpIndex )
#设置输入量和输出量
Scale_X = Data.iloc[:,:]
Scale_Y = Data.iloc[:,0]
#将数据处理到（-1，1）之间
scaler = MinMaxScaler(feature_range=(-1,1))
data = scaler.fit_transform(Scale_X)
y = np.array(Scale_Y)
y = scaler.fit_transform(y.reshape(-1,1))
#设置时间序列，时间步为100
time_steps = 100
X_new = np.zeros((data.shape[0] - time_steps + 1, time_steps, data.shape[1]))
y_new = np.zeros((y.shape[0] - time_steps + 1, 1))
for ix in range(X_new.shape[0]):
    for jx in range(time_steps):
        X_new[ix, jx, :] = data[ix + jx, :]
    y_new[ix] = y[ix + time_steps -1]
print(X_new.shape, y_new.shape)
'''这里采用自动分训练集和测试集，若采用sklearn的train_test_split会出错，原因在于这是随机分的，会打乱标签分布'''
split = int(0.8 * data.shape[0])
X_train = X_new[:split]
X_test = X_new[split:]

Y_train = y_new[:split]
Y_test = y_new[split:]

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
#将数据添加到计算图上
X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
Y_train = torch.from_numpy(Y_train).type(torch.Tensor)
Y_test = torch.from_numpy(Y_test).type(torch.Tensor)
#预测模型的构建
class net(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim,dropout=0.2):
        super(net,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
       # self.drop = nn.Dropout(dropout)
        #self.conv1d = nn.Conv1d()
        self.lstm = nn.LSTM(input_dim,hidden_dim,num_layers,batch_first = True,dropout=dropout)
        self.function = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        #x = self.drop(dt)
        #h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_().cuda()
        #c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_().cuda()
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_dim).requires_grad_()
        out,(hn,cn) = self.lstm(x,(h0.detach(),c0.detach()))
        #从out中获得最终输出的状态h
        out = self.function(out[:,-1,:])
        #out = out.squeeze(-1)
        return out
#设置超参数
input_dim = 4
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs =100

model = net(input_dim=input_dim,hidden_dim=hidden_dim,num_layers=num_layers,output_dim=output_dim)
optim = torch.optim.Adam(model.parameters(),lr = 0.01,weight_decay=0.01)

use_gpu = torch.cuda.is_available()
hist = np.zeros(num_epochs)
start_time = time.time()
print(model)
#训练
for i in range(num_epochs):
    if use_gpu:
        model = model.cuda()
        y_train_pred = model(X_train)
        loss = loss = nn.functional.mse_loss(y_train_pred,Y_train)
        loss = loss.cuda()
        print("Epoch",i+1,"MSE",loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()              
    else:
        y_train_pred = model(X_train)
        #设置损失函数为mse
        loss = loss = nn.functional.mse_loss(y_train_pred,Y_train)
        print("Epoch",i+1,"MSE",loss.item())
        hist[i] = loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
training_time = time.time()-start_time
torch.save(model.state_dict(),'model.pkl')
print("Training time:{}".format(training_time))
#绘制预测图形

import math,time
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
lstm = []
y_train_pred = model(X_train)
y_test_pred = model(X_test)
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(Y_train)
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(Y_test.detach().numpy())
memery_days = time_steps
original = scaler.inverse_transform(y.reshape(-1,1))
trainPredictPlot = np.empty_like(y)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[memery_days:len(y_train_pred)+memery_days,:] = y_train_pred
testPredictPlost = np.empty_like(y)
testPredictPlost[:,:] = np.nan
testPredictPlost[len(y_train_pred)-1:len(y_test_pred)+len(y_train_pred)-1,:] = y_test_pred
predictions = np.append(trainPredictPlot,testPredictPlost,axis=1)
predictions = np.append(predictions,original,axis=1)
result = pd.DataFrame(predictions)
import plotly.express as px
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index,y=result[0],mode='lines',name='Train Prediction')))
fig.add_trace(go.Scatter(x=result.index,y=result[1],mode='lines',name='Test Prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index,y=result[2],mode='lines',name='Actual Value')))
fig.update_layout(xaxis = dict(showline=True,showgrid=True,showticklabels=False,linecolor='white',linewidth=2),
                 yaxis = dict(title_text='Close (RMB)',titlefont=dict(family='Rockwell',color='white',),
                             showline=True,showgrid=True,showticklabels=True,linecolor='white',linewidth=2,
                             ticks='outside',tickfont=dict(family='Rockwell',color='white',),),
                 showlegend=True,template = 'plotly_dark')
annotations = []
annotations.append(dict(xref='paper',yref='paper',x=0.0,y=1.05,
                       xanchor='left',yanchor='bottom',text='Result(RNN)',
                       font=dict(family='Rockwell',color='white'),
                       showarrow=False))
fig.update_layout(annotations=annotations)
fig.show()

























