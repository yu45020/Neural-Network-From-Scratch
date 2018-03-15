from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.neural_network import MLPClassifier

from neural_net import *
import pickle
import matplotlib.pyplot as plt


with open('train_x', 'rb') as f:
    train_x = pickle.load(f)

with open('train_y', 'rb') as f:
    train_y = pickle.load(f)

with open('test_x', 'rb') as f:
    test_x = pickle.load(f)

with open('test_y', 'rb') as f:
    test_y = pickle.load(f)

list_activations = ['ReLu'] * 3 + ['sigmoid']
list_activations = ['ReLu'] * 3 + ['softmax']


X = train_x
Y = train_y

layers_dims = [100, 100,20, 1]
np.random.RandomState(1)
params=None
costs_total = []
params, costs = neural_net(X, Y, layers_dims,
               list_activations,
               learning_rate=5e-4,
               batch_iteration=1000,
               batch_size=256,
               trained_params=params,
               lambd=1e-2,
               batch_norm=True)


plt.plot(costs)
costs_total = np.append(costs_total, costs)
iteras = [i for i in range(len(costs_total)) if i % 100 == 0]
costs_ = costs_total[iteras]
plt.plot(iteras, costs_)



# sigmoid
train_y_pred = predict(train_x, params)
train_y_pred = np.int64(train_y_pred>0.5)
precision_score(train_y[0], train_y_pred[0])
recall_score(train_y[0], train_y_pred[0])
f1_score(train_y[0], train_y_pred[0])
np.mean(np.equal(train_y, train_y_pred))
np.average(train_y == train_y_pred)


test_y_pred = predict(test_x, params)
test_y_pred = np.int64(test_y_pred>0.5)
precision_score(test_y[0], test_y_pred[0])
recall_score(test_y[0], test_y_pred[0])
f1_score(test_y[0], test_y_pred[0])
np.average(test_y == test_y_pred)


## for softmax predict ###
train_y_pred = predict(train_x, params)

y_pred = np.int64(train_y_pred == train_y_pred.max(axis=0))
y_pred = train_y_pred.argmax(axis=0)
y_labels = train_y.argmax(axis=0)
np.mean(y_labels == y_pred)

y_test_pred = predict(test_x, params)
y_test_pred = y_test_pred.argmax(axis=0)
np.mean(y_test_pred==test_y.argmax(axis=0))

test_y_pred = predict(test_x, params)
test_y_pred = np.int64(test_y_pred>0.5)

precision_score(test_y[0], test_y_pred[0])
recall_score(test_y[0], test_y_pred[0])
f1_score(test_y[0], test_y[0])

np.mean(np.equal(test_y_pred, test_y))
np.average(test_y_pred == test_y)
##############################################
