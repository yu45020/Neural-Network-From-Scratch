from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from neural_net import *
import pickle
import matplotlib.pyplot as plt
from sklearn.externals import joblib
with open('mist.gz', 'rb') as f:
    dat = joblib.load(f)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standarscale = StandardScaler()


X_ = dat['data']
Y_ = dat['target']
X_, Y_ = shuffle(X_, Y_)

X = X_[:1000]
Y = Y_[:1000]

# +++++++++++++++++++++++++++++++++++++
#           Comparison to Scipy toolkit
# -------------------------------------
Logi = LogisticRegression(penalty='l2', C=1, max_iter=100)
Logi.fit(X, Y)
Logi.score(X_[2000:], Y_[2000:])
Logi.score(X, Y)

from sklearn.preprocessing import LabelBinarizer
LB = LabelBinarizer()
Y_train = LB.fit_transform(Y)
Y_test_ = LB.transform(Y_[2000:])

list_activations =['ReLu']*3 + ['softmax']
layers_dims = [100,100,50,10]
X_train = X.T
Y_train = Y_train.reshape((10, 1000))
Y_test = Y_test_.reshape((10, Y_test_.shape[0]))

np.random.RandomState(1)
params=None
costs_total = []
params, costs = neural_net(X_train/255, Y_train, layers_dims,
               list_activations,
               learning_rate=5e-5,
               batch_iteration=1000,
               batch_size=128,
               trained_params=params,
               lambd=0.004,
               batch_norm=True)

plt.plot(costs)

y_test_pred = predict(X_[2000:].T/255, params)
y_test_pred = np.int64(y_test_pred == y_test_pred.max(axis=0))
y_test_pred = y_test_pred.argmax(axis=0)
labels = Y_test.argmax(axis=0)
np.mean(y_test_pred==labels)

# +++++++++++++++++++++++++++++++++++++
#           Prediction
# -------------------------------------



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = standarscale.fit_transform(X_train)
X_test = standarscale.transform(X_test)



from sklearn.decomposition import PCA
pca = PCA(10)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.preprocessing import LabelBinarizer
LB = LabelBinarizer()

Y_train = LB.fit_transform(Y_train)
Y_test = LB.transform(Y_test)

X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T

list_activations = ['ReLu'] * 3 + ['softmax']

layers_dims = [100, 100, 50, Y_train.shape[0]]

np.random.RandomState(1)
params=None
costs_total = []
params, costs = neural_net(X_train, Y_train, layers_dims,
               list_activations,
               learning_rate=5e-4,
               batch_iteration=10,
               batch_size=256,
               trained_params=params,
               lambd=0,
               batch_norm=True)

plt.plot(costs)

## for softmax predict ###
train_y_pred = predict(X_train, params)
y_pred = np.int64(train_y_pred == train_y_pred.max(axis=0))
y_pred = train_y_pred.argmax(axis=0)
Y_train_label = Y_train.argmax(axis=0)
np.mean(Y_train_label == y_pred)

test_y_pred = predict(X_test, params)
test_y_pred = np.int64(test_y_pred == test_y_pred.max(axis=0))
test_y_pred = test_y_pred.argmax(axis=0)
Y_test_label = Y_test.argmax(axis=0)
np.mean(Y_test_label == test_y_pred)
##############################################

with open("mnist_params", 'wb') as f:
    pickle.dump([params.params, params.adam_params, params.list_activations], f)

