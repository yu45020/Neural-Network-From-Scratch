import numpy as np
from sklearn.utils import shuffle
from collections import namedtuple
from utils import *
from regularization import *
from activation_cost_fns import *
from batch_normolizer import *
from update_parameter_methods import *
from numba import jit

def init_layer_parameters(layer_dims):
    L = len(layer_dims)
    params = dict([[i, {}] for i in range(1, L)])
    for i in range(1, L):
        current, prev = layer_dims[i], layer_dims[i - 1]
        params[i]['W'] = np.random.randn(current, prev) * np.sqrt(2 / prev)
        params[i]['b'] = np.zeros((current, 1))
    return params

def forward_one_layer(A_prev, parameters, layer, activation_fn, predict_time):
    cache = {}

    W = parameters.params[layer]['W']
    b = parameters.params[layer]['b']
    Z_current = np.dot(W, A_prev) + b

    if layer in parameters.bn_params:
        Z_current = batch_normalizer(Z_current, parameters, layer, predict_time)

    A_current = activation_fn(Z_current)
    cache.update({"A": A_current, "Z": Z_current})
    return cache

def forward_propagation(X, parameters, activation_fns, predict_time=False):
    L = list(parameters.params)  # number of layers
    caches = dict([[i, {}] for i in L])

    caches[0] = {"A": X}
    A_prev = X

    for i in L:
        activation = get_activation_fn(activation_fns[i - 1])
        cache = forward_one_layer(A_prev, parameters, i, activation, predict_time)
        A_prev = cache['A']
        caches[i] = cache
    return caches

def backward_output_layer(Y, cache, A_prev,
                          parameters, layer,
                          loss_grad, regular_grad_fn,
                          regular_lambd):
    param = parameters.params[layer]
    m = Y.shape[1]
    dZ_current = loss_grad(Y, cache)
    dW = np.dot(dZ_current, A_prev.T) * 1. / m + regular_grad_fn(regular_lambd, param['W'], m)
    db = np.sum(dZ_current, axis=1, keepdims=True) * 1. / m
    grad = {'dW': dW, 'db': db}
    return dZ_current, grad

def backward_hidden_layer(dA_current, Z_current,
                          A_prev, parameters,
                          layer,gradient_fn,
                          regular_grad_fn, regular_lambd):
    # from layer [L-1, ... , 1]
    grad = {}
    m = Z_current.shape[1]
    dZ_current = dA_current * gradient_fn(Z_current)
    param = parameters.params[layer]
    if layer in parameters.bn_params:
        bn_param = parameters.bn_params[layer]
        dZ_current, dgamma_var, dbeta_mu = batch_normalizer_grad(dZ_current, Z_current, param, bn_param)
        grad.update({"dgamma_var": dgamma_var, 'dbeta_mu': dbeta_mu})

    dW = np.dot(dZ_current, A_prev.T) * 1. / m + regular_grad_fn(regular_lambd, param['W'], m)
    db = np.sum(dZ_current, axis=1, keepdims=True) * 1 / m

    grad.update({'dW': dW, 'db': db})
    return dZ_current, grad

def backward_propagation(Y, parameters, caches,
                         activation_fns,
                         regular_grad_fn,
                         regular_lambd):
    L = list(parameters.params)
    grads = dict([[i, {}] for i in L])

    layer = L[-1]  # last layer
    A_prev = caches[layer - 1]['A']
    loss_grad = get_loss_fn_grad_fn(activation_fns[-1])
    dZ_current, grad = backward_output_layer(Y, caches[layer], A_prev,
                                             parameters, layer,
                                             loss_grad, regular_grad_fn, regular_lambd)
    grads[layer].update(grad)

    for i in reversed(L[:-1]):  # skip the output layer
        dA_current = np.dot(parameters.params[i + 1]['W'].T, dZ_current)
        A_prev = caches[i - 1]['A']
        gradient_fn = get_activation_grad(activation_fns[i - 1])
        Z_current = caches[i]['Z']
        dZ_current, grad = backward_hidden_layer(dA_current, Z_current,
                                                 A_prev, parameters, i,
                                                 gradient_fn, regular_grad_fn,
                                                 regular_lambd)
        grads[i] = grad

    return grads


@timer_decorator
def neural_net(X, Y, layer_dims,
               list_activations,
               learning_rate,
               batch_iteration=1000,
               batch_size=128,
               trained_params=None,
               lambd=0,
               batch_norm=False,
               tol=1e-8):

    assert len(list_activations) == len(layer_dims)
    assert layer_dims[-1] == layer_dims[-1]

    layer_dims = [X.shape[0]] + layer_dims
    L = len(list_activations)

    if trained_params is None:
        # Parameters = namedtuple('Parameters', ['params', 'bn_params', 'adam_params', 'list_activations'])
        parameters =DataContainter(params={}, grads={}, bn_params={}, adam_params={}, list_activations=list_activations)

        parameters.params = init_layer_parameters(layer_dims)

        if batch_norm:
            parameters.bn_params = init_batch_norm_params(layer_dims, parameters.params)

        parameters.adam_params = init_adam_update_parameters(parameters.params)

    else:
        parameters = trained_params

    costs = []
    cost_old = np.inf
    cost_diff = np.inf

    compute_loss_fn = get_loss_fn(list_activations[-1])

    regularization_grad = regularization_l2_grad
    regularization_fn = regularization_l2
    n = 0
    t = 0
    while n < batch_iteration and cost_diff > tol:
        X, Y = shuffle(X.T, Y.T)
        X, Y = X.T, Y.T
        batch_cost = 0
        for X_batch, Y_batch in split_batches(X, Y, batch_size):
            t += 1
            caches = forward_propagation(X_batch, parameters,
                                                    list_activations)
            parameters.grads = backward_propagation(Y_batch, parameters, caches,
                                         list_activations, regularization_grad, lambd)

            cost = compute_loss_fn(Y_batch, caches[L]['A']) + regularization_fn(lambd, parameters.params, Y_batch.shape[1])
            # parameters = update_parameters_adam(parameters, grads, learning_rate, t)
            parameters.params = update_parameters_gds(parameters.params, parameters.grads, learning_rate)
            batch_cost += cost / Y_batch.shape[1]
            # costs.append(cost / Y_batch.shape[1])

        cost_diff = abs(cost_old - batch_cost)
        cost_old = batch_cost

        costs.append(batch_cost)
        if n % 100 == 0:
            print("Current {} iterations cost {}".format(n, batch_cost))
        n += 1
    print("Final {} interations cost {}".format(n, batch_cost))


    return parameters, np.array(costs)


def predict(X, parameters):
    list_activations = parameters.list_activations
    L = list(parameters.params)[-1]
    caches= forward_propagation(X, parameters, list_activations, predict_time=True)
    A_final = caches[L]['A']
    return A_final

