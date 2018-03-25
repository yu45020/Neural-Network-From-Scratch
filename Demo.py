from recurrent_neural_net import *
import pandas as pd
import matplotlib.pyplot as plt


dat = pd.read_csv('VXOCLS.csv', na_values='.')
dat['DATE'] = pd.to_datetime(dat['DATE'])
dat['dateofweek'] = dat['DATE'].dt.dayofweek  # Monday=0, Sunday=6
A = dat.loc[dat.dateofweek == 0]
B = dat.loc[dat.dateofweek == 1]
first_date = A.iloc[0].DATE
last_date = B.iloc[-1].DATE

dat.fillna(method='ffill', inplace=True)

dat = dat[(first_date <= dat.DATE) & (dat.DATE <= last_date)]
dat = dat.reset_index(drop=True)
dat.tail(3)
data = dat.VXOCLS.values / 100
train, test = data[:7000], data[7000:]
X_ = train[:-1]
test_x = test[:-1]
Y_ = train[1:]
test_y = test[1:]
m = len(X_) // 5
m_test = len(test_x)//5
Tx = 1
nx = 1
X = np.zeros((nx, m, Tx))
Y = np.zeros((nx, m, Tx))
X_test = np.zeros((nx, m_test, Tx))
Y_test = np.zeros((nx, m_test, Tx))
tx = 0
for i in range(m):
    X[:,i,:] = X_[tx:tx+Tx]
    Y[:,i,:] = Y_[tx:tx+Tx]
    tx +=Tx

tx = 0
for i in range(m_test):
    X_test[:,i,:] = test_x[tx:tx+Tx]
    Y_test[:,i,:] = test_y[tx:tx+Tx]
    tx += Tx

params = None
params, costs = rnn(X, Y, 30, 1, parameters=params, batch_size=32,
                    activation_input=tanh, activation_output=linear_identity, loss=mean_square,
                    learning_rate=5e-4, max_iteration=1000)

plt.plot(costs)
cell_size_a = params['Waa'].shape[0]
a_0 = np.zeros((cell_size_a, X.shape[1]))
caches = forward_propagation(a_0, X, params, tanh, linear_identity)
y_pred = []
y_act = []
for i in range(m):
    y_m = [caches[j]['y_current'].squeeze()[i] for j in range(1, Tx+1)]
    y_pred +=y_m
    y_act += np.ndarray.tolist(Y[:,i,:].squeeze())

plt.plot(y_pred, color='red')
plt.plot(y_act, color='black')
plt.plot(Y_, color='green')


a_0 = np.zeros((cell_size_a, X.shape[1]))
caches_train = forward_propagation(a_0, X, params, tanh, linear_identity)
a_current = np.array(caches_train[Tx]['a_current'])
a_current = a_current.mean(axis=1, keepdims=True)
x_current = test_x[0]
y_pred_test = []
for i in range(5):
    cache, a_current = forward_one_cell(a_current, x_current, params, tanh, linear_identity)
    x_current = cache['y_current'][0][0]
    y_pred_test.append(x_current)

plt.plot(y_pred_test)
plt.plot(test_y[:5])