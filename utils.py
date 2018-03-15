import time

import numpy as np
from functools import wraps

class DataContainter(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self.__dict__ = self

    def __getstate__(self):
        return str(self.keys())

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def flatten_dic(dict):
    out = [x for d in iter(dict.values()) for y in iter(d.values()) for x in y]
    out = np.concatenate(out)
    return np.ndarray.flatten(out)

def split_batches(X, Y, batch_size):
    # assume X = (Nx , m) where Nx features and m  samples
    m = X.shape[1]
    for n in range(0, m, batch_size):
        yield X[:, n:min(batch_size+n, m)], Y[:, n:min(batch_size+n, m)]


def timer_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_t = time.time()
        result = fn(*args, **kwargs)
        print("Runtime: {}".format(time.time() - start_t))
        return result
    return wrapper


def neural_net_cost_logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        cost = fn(*args, **kwargs)
        if wrapper.calls % 100 == 0:
            print("Cost after {} iterations : {:.7f}".format(wrapper.calls, cost))
        return cost
    wrapper.calls = 0
    return wrapper

def dropout_layer(Z_current, keep_prop, cache):
    # not recommended when used batch norm
    mask = np.random.rand(*Z_current.shape) < keep_prop
    Z_current = Z_current * mask / keep_prop
    cache.update({'mask': mask, "keep_prop": keep_prop})
    return Z_current
