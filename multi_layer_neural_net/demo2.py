from sklearn.preprocessing import LabelBinarizer, StandardScaler
from recurrent_neural_net import *
import pandas as pd
import matplotlib.pyplot as plt


encoder = LabelBinarizer()
scaler = StandardScaler()

dat = pd.read_csv('beijing_pm2.5.csv', index_col=0)
dat['pm2.5_pred'] = dat['pm2.5'].shift(-1)
dat.dropna(axis=0, inplace=True)
dat.reset_index(drop=True, inplace=True)
dat.head(5)
dat.shape
cbwb = encoder.fit_transform(dat[['cbwd']])
dat = dat[["pm2.5_pred",'pm2.5','DEWP','TEMP', 'PRES','Iws','Is','Ir']]
dat = pd.concat([dat, pd.DataFrame(cbwb)], axis=1)
m = 4000
mx = dat.shape[0]-m
train_x = dat.iloc[:m,1:]
train_y = dat.iloc[:m, 0]
train_x[['DEWP', 'TEMP','PRES']] = scaler.fit_transform(train_x[['DEWP', 'TEMP','PRES']] )
test_x = dat.iloc[m:,1:]
test_x[['DEWP', 'TEMP','PRES']] = scaler.transform(test_x[['DEWP', 'TEMP','PRES']] )

test_y = dat.iloc[m:, 0]

Tx = 1
nx = 11
X_, Y_ = train_x.values.T, train_y.values.reshape(1,m)
X = np.zeros((nx, m, Tx))
Y = np.zeros((1, m, Tx))
tx = 0
for i in range(m):
    X[:,i,:] = X_[:,tx:(tx+Tx)]
    Y[:,i,:] = Y_[:,tx:(tx+Tx)]
    tx +=Tx

X_test = np.zeros((nx, mx, Tx))
Y_test = np.zeros((1, mx, Tx))
X_, Y_ = test_x.values.T, test_y.values.reshape(1,mx)
tx = 0
for i in range(mx):
    X_test[:,i,:] = X_[:,tx:tx+Tx]
    Y_test[:,i,:] = Y_[:,tx:tx+Tx]
    tx += Tx


params = None
params, costs = rnn(X, Y, 50, 1, parameters=params, batch_size=64,
        activation_input=tanh, activation_output=linear_identity,loss=mean_square,
        learning_rate=5e-4, max_iteration=1000)

plt.plot(costs)
a_0 = np.zeros((50, X.shape[1]))
caches = forward_propagation(a_0, X, params, tanh, linear_identity)
y_pred = caches[1]['y_current'].squeeze()
plt.plot(y_pred, color='green')
plt.plot(train_y, color='red')
