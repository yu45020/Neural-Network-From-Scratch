"""
Basic RNN
one to one framework


"""
from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray

from utils import *


def forward_one_cell(a_prev, x_current, parameters, activation_input, activation_output):
    Waa, Wax, ba = parameters['Waa'], parameters['Wax'], parameters['ba']
    Wya, by = parameters['Wya'], parameters['by']

    a = np.dot(Waa, a_prev) + np.dot(Wax, x_current) + ba
    a_current = activation_input.forward(a)

    y = np.dot(Wya, a_current) + by
    y_current = activation_output.forward(y)

    cache = {'a': a, 'a_current': a_current, 'y': y, 'y_current': y_current}
    return cache, a_current


def forward_propagation(a_0, X, parameters, activation_linear, activation_y):
    Tx = X.shape[2]
    a_current = a_0
    caches = {i: {} for i in range(Tx + 1)}
    caches[0]['a_current'] = a_0

    for i in range(Tx):
        cache, a_current = forward_one_cell(a_current, X[..., i], parameters,  activation_linear, activation_y)
        caches[i + 1] = cache
    return caches


def back_one_cell(forward_caches, Tx, da_next, X, Y, m,
                  parameters, loss, activation_linear):
    dy_pred = loss.backward(forward_caches[Tx]['y_current'], Y[..., Tx - 1])
    dby = dy_pred  # * activation_y.backward(forward_caches[Tx]['y'], Y[..., Tx])
    dWya = np.dot(dby, forward_caches[Tx]['a_current'].T)
    da_current = np.dot(parameters['Wya'].T, dby) + da_next

    dba = da_current * activation_linear.backward(forward_caches[Tx]['a'])
    dWaa = np.dot(dba, forward_caches[Tx - 1]['a_current'].T)
    dWax = np.dot(dba, X[..., Tx - 1].T)
    da_prev = np.dot(dWaa.T, dba)

    grad = {'Wya': dWya / m, 'by': dby.mean(axis=1, keepdims=True),
            'Waa': dWaa / m, 'ba': dba.mean(axis=1, keepdims=True),
            'Wax': dWax / m}
    return grad, da_prev


def back_propagation(forward_caches, X, Y, parameters, loss, activation_linear):
    _, m, Ty = Y.shape
    grads = {k: 0 for k in parameters.keys()}
    da_current = np.zeros_like(forward_caches[0]['a_current'])
    for i in reversed(range(1, Ty + 1)):
        grad, da_current = back_one_cell(forward_caches, i, da_current, X, Y, m,
                                         parameters, loss, activation_linear)
        grads = {k: grads.get(k) + grad.get(k) for k in grad.keys()}
    return grads


def compute_loss(forward_cache, Y, loss_fn):
    loss = np.array([loss_fn.forward(forward_cache[Tx]['y_current'], Y[..., Tx-1]) for Tx in range(1, Y.shape[2]+1)])
    return np.sum(loss/Y.shape[1])  # Y.shape(n_y, m, Ty)


def init_params(n_x, n_a, n_y):
    parameters = {'Wya': np.random.randn(n_y, n_a) * 0.01,
                  'by': np.zeros((n_y, 1)),
                  'Waa': np.random.randn(n_a, n_a) * 0.01,
                  'ba': np.zeros((n_a, 1)),
                  'Wax': np.random.randn(n_a, n_x) * 0.01}
    return parameters


def rnn(X, Y, cell_size_a, cell_size_y, parameters=None, batch_size=16,
        activation_input=tanh, activation_output=softmax,loss=cross_entropy,
        learning_rate=5e-4, max_iteration=1000):
    # X.shape ( nx, m, Tx )  Y.shape (ny, m, Ty)
    X_size, m, Tx = X.shape
    Y_size, m, Ty = Y.shape

    if parameters is None:
        parameters = init_params(X_size, cell_size_a, cell_size_y)
    optimizer = OptimizationAdam
    adam_params = optimizer.init_params(parameters)
    t = 1
    n = 0
    costs = []
    while n <= max_iteration:
        batch_cost = 0
        for X_batch, Y_batch in split_batches(X,Y,batch_size):
            a_0 = np.zeros((cell_size_a, X_batch.shape[1]))
            caches = forward_propagation(a_0, X_batch, parameters, activation_input, activation_output)
            grads = back_propagation(caches, X_batch, Y_batch, parameters, loss, activation_input)
            parameters, adam_params = optimizer.update_params(parameters, grads, adam_params, learning_rate, t)
            batch_cost += compute_loss(caches, Y_batch, loss)
        costs.append(batch_cost)
        if n % 100 == 0:
            print("Iterations {} cost {}".format(n, batch_cost))
            t += 1
        n += 1
    return parameters, costs

def rnn_predict(parameters,X, activation_input=tanh, activation_output=softmax):
    cell_size_a = parameters['Waa'].shape[0]
    a_0 = np.zeros((cell_size_a, X.shape[1]))
    caches = forward_propagation(a_0, X, parameters, activation_input, activation_output)
