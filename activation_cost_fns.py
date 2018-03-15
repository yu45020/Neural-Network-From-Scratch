"""
Contain activation functions, cost functions, and their gradients
"""
from numba import jit
import numpy as np


# +++++++++++++++++++++++++++++++++++++
#           Activation Functions
# -------------------------------------
#@jit(nopython=True)
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
#@jit(nopython=True)
def sigmoid_grad(Z):
    s = sigmoid(Z)
    return s * (1 - s)

#@jit(nopython=True)
def ReLu(Z):
    return np.maximum(0, Z)

#@jit(nopython=True)
def ReLu_grad(Z):
    return np.int64(Z > 0)

#@jit(nopython=True)
def soft_max(Z):
    # assume Z(Ny, m) where Ny are classes and m are sample sizes
    Z_ = Z - Z.max(axis=0)
    return np.exp(Z_) / np.sum(np.exp(Z_), axis=0)

def cosh(Z):
    return np.cosh(Z)

def cosh_grad(Z):
    return np.sinh(Z)

def tanh(Z):
    return np.tanh(Z)

def tanh_grad(Z):
    return 1-tanh(Z)**2

def linear_identity(Z):
    return Z


#@jit(nopython=True)
def get_activation_fn(fn_name):
    if fn_name == 'sigmoid':
        return sigmoid
    elif fn_name == 'ReLu':
        return ReLu
    elif fn_name == 'softmax':
        return soft_max
    elif fn_name =='cosh':
        return cosh
    elif fn_name == 'tanh':
        return tanh
    elif fn_name == 'linear':
        return linear_identity
    else:
        raise Exception("The activation function is not defined: {}".format(fn_name))



#@jit(nopython=True)
def get_activation_grad(fn_name):
    if fn_name == 'sigmoid':
        return sigmoid_grad
    elif fn_name == 'ReLu':
        return ReLu_grad
    elif fn_name =='cosh':
        return cosh_grad
    elif fn_name =='tanh':
        return tanh_grad
    else:
        raise Exception("The activation function's gradient is not defined: {}".format(fn_name))


# +++++++++++++++++++++++++++++++++++++
#           Cost Functions
# -------------------------------------

#@jit(nopython=True)
def softmax_loss(Y, A_final):
    # A_final.argx(axis=0)
    # y_pred = np.int64(A_final == A_final.max(axis=0))
    return -np.sum(Y * np.log(A_final))

#@jit(nopython=True)
def soft_max_loss_grad(Y, cache):
    # cache = {A, Z} from forward prop
    return cache['A'] - Y  # dZ

#@jit(nopython=True)
def sigmoid_cross_entropy_loss(Y, A_final):
    loss = Y * np.log(A_final) + (1 - Y) * np.log(1 - A_final)
    return -np.sum(loss)

#@jit(nopython=True)
def sigmoid_cross_entropy_loss_grad(Y, cache):
    A_final, Z = cache['A'], cache['Z']
    dA = -(np.divide(Y, A_final) - np.divide(1 - Y, 1 - A_final))
    return dA * sigmoid_grad(Z) # dZ

def mean_square_loss(Y, A_final):
    return np.sum(np.square(A_final-Y))

def mean_square_loss_grad(Y, cache):
    return 2*(cache['A'] - Y)


#@jit(nopython=True)
def get_loss_fn(fn_name):
    if fn_name == 'softmax':
        return softmax_loss
    elif fn_name == 'sigmoid':
        return sigmoid_cross_entropy_loss
    elif fn_name =='linear':
        return mean_square_loss
    else:
        raise Exception("The loss function is not defined: {}".format(fn_name))


#@jit(nopython=True)
def get_loss_fn_grad_fn(fn_name):
    if fn_name == 'softmax':
        return soft_max_loss_grad
    elif fn_name == 'sigmoid':
        return sigmoid_cross_entropy_loss_grad
    elif fn_name =='linear':
        return mean_square_loss_grad
    else:
        raise Exception("The loss function's gradient is not defined: {}".format(fn_name))

