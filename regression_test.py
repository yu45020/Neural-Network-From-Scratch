from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.neural_network import MLPClassifier

from neural_net import *
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



import pandas as pd
dat = pd.read_csv("df1ema.csv")
dat.drop('Unnamed: 0', axis=1, inplace=True)
dat['DATE'] = pd.to_datetime(dat['DATE'])
dat.index = dat['DATE']

start_date = dat.index.min() - pd.DateOffset(day=1)
end_data = dat.index.max()
dates = pd.date_range(start_date, end_data, freq='D')
dat = dat.reindex(dates)

dat2 = dat.interpolate(method='cubic', limit_direction='both')
dat2 = dat
dat2 = dat2.drop('DATE', axis=1)
m_train = int(len(dat2)*0.8)
X_train, X_test = dat2[:m_train], dat2[m_train:]
Y_train = dat2[1:m_train]['DICS']
Y_test = dat2[(1+m_train):]['DICS']
X_train = X_train[:-1]
X_test = X_test[:-1]

plt.plot(Y_train)
plt.plot(Y_test)

list_activations = ['cosh', 'tanh', 'linear']
layers_dims = [1,1, 1]



X = X_train.values
X = X.T
Y = Y_train.values
Y = Y.reshape((1, Y.shape[0]))

X_test= X_test.values
X_test = X_test.T
Y_test = Y_test.values
Y_test = Y_test.reshape((1, Y_test.shape[0]))

n = 100
X = np.random.randn(4,n)*10

w = np.array([20,30, -1, -9])
b = 100
Y = np.dot(w, X) + b + np.random.randn(n)
Y = Y.reshape(1,n)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X.T, Y.T)
lr.coef_
lr.intercept_
lr.score(X.T, Y.T)
np.random.RandomState(1)
params=None
costs_total = []
list_activations = ['linear']
layers_dims = [1]
params, costs = neural_net(X, Y, layers_dims,
               list_activations,
               learning_rate=1,
               batch_iteration=2000,
               batch_size=256,
               trained_params=params,
               lambd=0,
               batch_norm=False)
params.params
plt.plot(costs)
y_train_pred = predict(X, params)
r2_score(Y[0], y_train_pred[0])

y_test_pred = predict(X_test, params)
r2_score(Y_test[0], y_test_pred[0])

Y_actual = np.concatenate([Y[0], Y_test[0]])
Y_p = np.concatenate([y_train_pred[0], y_test_pred[0]])
plt.plot(Y_actual, color='red')
plt.plot(Y_p, color='black')
len(dat2)
len(Y_p)
params.params[1]['W']
list(dat2)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X.T, Y.T)


A = pd.DataFrame(list(dat2))
A['coef_1'] = params.params[1]['W'][0]
A['linear'] = lr.coef_[0]

lr.score(X.T,Y.T)
lr.score(X_test.T, Y_test.T)
lr_r = lr.predict(X_test.T)
plt.plot(lr_r)
plt.plot(Y_test[0])
plt.plot(y_test_pred[0])