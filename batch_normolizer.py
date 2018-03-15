import numpy as np


# +++++++++++++++++++++++++++++++++++++
#           Batch Norm
# -------------------------------------

def init_batch_norm_params(layer_dims, params):
    L = len(layer_dims)

    bn_params = dict([[i, {}] for i in range(1, L-1)])
    for i in range(1, (L - 1)):  # skip the output layer
        current, prev = layer_dims[i], layer_dims[i - 1]
        if 'gamma_var' not in params[i]:  # add variables
            params[i]['gamma_var'] = np.random.randn(current, 1) * np.sqrt(1 / prev)
            params[i]['beta_mu'] = np.random.randn(current, 1) * np.sqrt(1 / prev)
        bn_params[i]['layer_mean'] = 0
        bn_params[i]['layer_var'] = 0
    return bn_params


def batch_normalizer(Z_current, parameters, layer, predict_time, epsilon=1e-8):
    # batch mean and var are update by v_mu_t = 0.9 v_mu_t-1 + 0.1 mu_t
    # Z_hat = Z_standardized * gamma + beta
    # Z_standardized = (Z_current - Z_mean) / sqrt(Z_var + epsilon)
    bn_param = parameters.bn_params[layer]
    param = parameters.params[layer]
    if predict_time:
        mu = bn_param['layer_mean']
        sd = bn_param['sd']

    else:  # when training
        mu = np.mean(Z_current, axis=1, keepdims=True)
        var = np.var(Z_current, axis=1, keepdims=True)
        bn_param['layer_mean'] = 0.9 * bn_param['layer_mean'] + 0.1 * mu
        bn_param['layer_var'] = 0.9 * bn_param['layer_var'] + 0.1 * var
        sd = np.sqrt((var + epsilon))
        de_mu = Z_current-mu
        bn_param.update({'sd': sd, 'de_mu': de_mu})

    Z_norm = (Z_current - mu) / sd
    Z_hat = param['gamma_var'] * Z_norm + param['beta_mu']
    parameters.bn_params[layer] = bn_param
    return Z_hat


def batch_normalizer_grad(dZ_current, Z_hat, param, bn_param):
    # Z_current = gamma * Z_hat + beta
    # Z_hat = (Z - mean(Z))/(sqrt(var(Z) + epsilon))

    layer_sd = bn_param['sd']
    Z_mean = bn_param['de_mu']  # Z - mu
    m = dZ_current.shape[1]

    dZ_hat = dZ_current * param['gamma_var']

    dvar = dZ_hat * Z_mean * (-1 / (2 * layer_sd ** 3))
    dvar = np.sum(dvar, axis=1, keepdims=True)

    dmu = np.sum(dZ_hat * (-1 / layer_sd), axis=1, keepdims=True)
    dmu += dvar * (-2) * np.sum(Z_mean, axis=1, keepdims=True)/m

    dZ = dZ_hat / layer_sd + dvar * (2 * Z_mean / m) + dmu / m
    dgamma_var = np.sum(dZ_current * Z_hat, axis=1, keepdims=True)
    dbeta_mu = np.sum(dZ_current, axis=1, keepdims=True)
    return dZ, dgamma_var, dbeta_mu
