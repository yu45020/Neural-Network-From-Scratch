# Neural Network From Scratch
TODO: 
1. Add regression, negative correlation with ensemble networks ("Ensemble learning via negative correlation"), etc 
2. Learn to write easy to read structure and class design. Orz. 

The math for basic frame work are in __init__ file, and more details will be updated. 

The model is built by numpy only, so it has all the advantages from numpy except the support for GPU.

## Model Details:
1. Basic multi-layer neural work with L2 regularization.
2. Activation functions are ReLu, Sigmoid, Softmax, and support classification.
3. Batch norm for each hidden layer.
4. Default optimization is Adam.

## Sample usage
Please see mnist_test.py

Main usage
```
X_train.shape  # (N_features, m_samples)
Y_train.shape  # (N_classes, m_samples)
list_activations = ['ReLu'] * 3 + ['softmax']
layers_dims = [100, 100, 50, Y_train.shape[0]]
params=None
params, costs = neural_net(X_train, Y_train, layers_dims,
               list_activations,
               learning_rate=5e-4,
               batch_iteration=10,
               batch_size=256,
               trained_params=params,  # use pre-trained values if needed
               lambd=0,     # l2 regularization
               batch_norm=True)
```