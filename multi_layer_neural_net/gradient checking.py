from activation_cost_fns import *
import numpy as np
from neural_net import *
import copy
m=20
X = np.random.randint(0, 1, (10, m))*10
Y = np.random.randint(0, 4, (4,m))
layers_dims = [10, Y.shape[0]]
list_activations = ['ReLu','softmax']
np.random.RandomState(1)
parameters =DataContainter(params={}, grads={}, bn_params={}, adam_params={}, list_activations=list_activations)
parameters.params = init_layer_parameters(layers_dims)

caches = forward_propagation(X, parameters,
                             list_activations)
L = len(list_activations)
parameters.grads = backward_propagation(Y, parameters, caches, list_activations, regularization_l2_grad, 0)


def gradient_check(X, Y, parameters, eps=1e-4):

    params = parameters.params
    grads = parameters.grads
    compute_loss_fn = get_loss_fn(parameters.list_activations[-1])
    results = copy.copy(params)

    for layer in params:
        for key in params[layer].keys():
            for i in range(params[layer][key].shape[0]):
               for j in range(params[layer][key].shape[1]):
                   params[layer][key][i][j] += eps
                   A_plus = predict(X, parameters)
                   cost_plus = compute_loss_fn(Y, A_plus)  * 1. / X.shape[1]
                   params[layer][key][i][j] -= 2. * eps
                   A_min = predict(X, parameters)
                   cost_mins = compute_loss_fn(Y, A_min)  * 1. / X.shape[1]
                   params[layer][key][i][j] += eps
                   grad_approx = (cost_plus - cost_mins) / (2. * eps)
                   results[layer][key][i][j] = grad_approx

    grads_flat = flatten_dic(grads)
    res_flat = flatten_dic(results)
    upper = np.linalg.norm(grads_flat-res_flat)
    lower = np.linalg.norm(res_flat) + np.linalg.norm(grads_flat)
    return upper/lower



check = gradient_check(X, Y, parameters, eps=1e-4)



# +++++++++++++++++++++++++++++++++++++
#           Activation fns
# -------------------------------------

eps = 1e-7
np.random.seed(1)
Y_ = np.random.randint(0, 6, 20)
Y = np.zeros((6,20))
for i in range(Y.shape[1]):
    Y[Y_[i],i] = 1

Z = np.random.randn(6,20)
A = soft_max(Z)
loss = softmax_loss(Y, A)
grad = soft_max_loss_grad(Y, {"A":A, "Z":Z})
approx_grad = np.zeros_like(grad)

for i in range(grad.shape[0]):
    for j in range(grad.shape[1]):
        Z_p = np.copy(Z)
        Z_p[i][j] += eps
        A_p = soft_max(Z_p)
        loss_p = softmax_loss(Y, A_p)

        Z_m = np.copy(Z)
        Z_m[i][j] -= eps
        A_m = soft_max(Z_m)
        loss_m = softmax_loss(Y, A_m)
        approx_grad[i][j] = (loss_p - loss_m) / (2 * eps)

upper = np.linalg.norm(approx_grad-grad)
lower = max(np.linalg.norm(approx_grad), np.linalg.norm(grad))
upper/lower