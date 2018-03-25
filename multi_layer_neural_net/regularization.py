import numpy as np

def regula_none(*args, **kwargs):
    return 0


def regularization_l2(lambd, params, m):
    # update all layers' parameters at once

    if lambd > 0:
        w = [params[i]['W'] for i in params.keys()]
        w_sum = list(map(lambda x: np.sum(np.square(x)), w))
        w_sum = np.sum(w_sum) * lambd / (2 * m)
    else:
        w_sum = 0
    return w_sum

def regularization_l2_grad(lambd, param_w, m):
    # update per layer
    if lambd > 0:
        return lambd * abs(param_w) / m
    else:
        return 0