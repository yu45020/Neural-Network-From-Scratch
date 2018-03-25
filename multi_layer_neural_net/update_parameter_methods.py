import numpy as np



def update_parameters_gds(params, grads, learning_rates):
    for i in list(params):  # update by each layer
        param, grad = params[i], grads[i]
        param = {k: v - learning_rates * grad['d' + k] for k, v in param.items()}
        params[i] = param
    return params


def init_adam_update_parameters(params):
    L = params.keys()
    adam_params = {i: {} for i in L}
    for i in L:
        adam_params[i]['adam_v'] = {k: np.zeros_like(v) for k, v in params[i].items()}
        adam_params[i]['adam_s'] = {k: np.zeros_like(v) for k, v in params[i].items()}
    return adam_params


def update_parameters_adam(parameters, grads, learning_rate,
                           t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for i in list(parameters.params):
        param = parameters.params[i]
        adam_param = parameters.adam_params[i]
        grad= grads[i]
        adam_param['adam_v'] = {k: beta1 * adam_param['adam_v'][k] + (1 - beta1) * grad["d" + k] for k in param.keys()}
        adam_param['adam_s'] = {k: beta2 * adam_param['adam_s'][k] + (1 - beta2) * grad['d' + k] ** 2 for k in
                                param.keys()}

        learning = (learning_rate * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
        dgrad = {k: v - learning * adam_param['adam_v'][k] / (np.sqrt(adam_param['adam_s'][k]) + epsilon) for k, v in
                 param.items()}

        parameters.params[i] = dgrad
        parameters.adam_params[i] = adam_param
    return parameters

