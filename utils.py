import numpy as np


class OptimizationAdam:
    @staticmethod
    def init_params(parameters):
        adam_params = {k: {'adam_v': np.zeros_like(parameters[k]),
                           'adam_s': np.zeros_like(parameters[k])} for k in parameters.keys()}
        return adam_params
    @staticmethod
    def update_params(parameters, grads, adam_params, learning_rate,
                      t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for i in parameters.keys():
            adam_params[i]['adam_v'] = beta1 * adam_params[i]['adam_v'] + (1 - beta1) * grads[i]
            adam_params[i]['adam_s'] = beta2 * adam_params[i]['adam_s'] + (1 - beta2) * grads[i] ** 2

            learning = (learning_rate * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
            parameters[i] -= learning * adam_params[i]['adam_v'] / (np.sqrt(adam_params[i]['adam_s']) + epsilon)

        return parameters, adam_params


class tanh:
    @staticmethod
    def forward(Z):
        return np.tanh(Z)
    @staticmethod
    def backward(Z):
        return 1 - np.tanh(Z) ** 2


class softmax:
    @staticmethod
    def forward(Z):
        Z_ = Z - Z.max(axis=0)
        return np.exp(Z_) / np.sum(np.exp(Z_), axis=0)
    @staticmethod
    def backward(Z, Y):
        return -np.sum(np.dot(np.log(Z), Y))


class cross_entropy:
    @staticmethod
    def forward(Y_pred, Y):
        # A_final.argx(axis=0)
        # y_pred = np.int64(A_final == A_final.max(axis=0))
        return -np.sum(np.log(Y_pred) * Y)
    @staticmethod
    def backward(Y_pred, Y):
        return Y_pred - Y  # dZ

class mean_square:
    @staticmethod
    def forward(Y_pred, Y):
        return np.sum(np.square(Y_pred - Y))
    @staticmethod
    def backward(Y_pred, Y):
        return 2. * (Y_pred - Y)

class linear_identity:
    @staticmethod
    def forward(Z):
        return Z
    @staticmethod
    def backward(Z, Y):
        return 1.


def split_batches(X, Y, batch_size):
    # assume X = (Nx , m, Tx) where Nx features and m  samples
    m = X.shape[1]
    for n in range(0, m, batch_size):
        yield X[:, n:min(batch_size + n, m), :], Y[:, n:min(batch_size + n, m),:]
