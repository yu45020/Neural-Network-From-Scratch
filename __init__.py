"""
Neural Network

# +++++++++++++++++++++++++++++++++++++
#           Basic Setup
# -------------------------------------
params:
    X: input data with shape (Nx, m), where Nx is feature sizes and m is sample sizes
    Y: output data with shape (Ny, m)
    W: parameters (n_current, n_prev) where n_current is # of current layer's neurons and n_prev is previous
    b: bias constant term (1, n_current)

forward propagation:
    Z = <W, X> + b
    A = f(Z)

Loss = L(Y, A_final), cost = mean(Loss)

backward propagation:
    output layer: (L)
       dL/dA_final = dLoss/dA_final
       dZ = dA * f'(Z)
       dW = <dZ, A.T> / m
       db = dZ / m

    hidden layers: (L-1, ..., 1)
        dA_l = <W_l+1.T, dZ_l+1>
        dZ_l = dA_l * f_i'(Z_l)
        dW_l = <dZ_l , A_l-1.T> / m
        db_l = dZ_l / m

Batch Norm:
Reference: Ioffe, Szegedy. "Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift"
"""

