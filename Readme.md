Foobar is a Python library for dealing with word pluralization.

## Dependency Installation
```bash
pip install -r requirement.txt
```
## Definition
### X_train, Y_train:
```
## Pick n_train samples as the training set for neural networks
train_id = np.random.choice(nN, size = n_train, replace = False)

## Use the rest for test
test_id = np.setdiff1d(range(nN), train_id)

## Tag = index of the variable to be tested for significance
X_tag = X_mat[tag,:].reshape((1, nN))
X_train  = X_tag[:,train_id]
Y_train = Y[:,train_id]
Y_test  = Y[:,test_id]
```

### mse_test_nn: mean square error for test data using sample mean
```
mse_test_nn = np.mean(np.power(Y_test - np.mean(Y_test), 2))
```

### n_h: number of hidden units for shallow neural network
```
n_h = np.floor(np.power(nN, deg)).astype(int)
```

### Function to get the mean square error for train data and y prediction for shallow neural network
```
mse_train_nn, predict_nn = fit_shallow_relu_nn(X_train, Y_train, n_h,
                                                optimizer = optimizer, epochs = epochs_nn,
                                                batch_size = batch_size,
                                                early_stopping = early_stopping,
                                                validation_split = validation_split,
                                                patience = patience_nn,
                                                min_delta = min_delta,
                                                drop_rate = drop_rate,
                                                verbose = verbose)
```
### Number of layers and nodes for each hidden layer 
``` 
num_layers = np.floor(np.power(nN, deg)).astype(int)
num_nodes = np.repeat(max_hidden_unit, num_layers)
```


### Function to get the mean square error for train data and y prediction for deep neural network
``` 
mse_train_dnn, predict_dnn = fit_deep_relu_nn(X_train, Y_train, num_nodes, num_layers,
                                              optimizer = optimizer, epochs = epochs_nn,
                                              batch_size = batch_size,
                                              early_stopping = early_stopping,
                                              validation_split = validation_split,
                                              patience = patience_nn,
                                              min_delta = min_delta,
                                              drop_rate = drop_rate,
                                              verbose = verbose)
```

### Calculate kappa in the manuscript
``` 
def calculate_kappa(Y_train, predict_train,
                    plus = False):
    if plus:
        kappa = np.mean(np.power(Y_train - predict_train.T, 4))
    else:
        mse_train = np.mean(np.power(Y_train - predict_train.T, 2))
        kappa = np.mean(np.power(Y_train - predict_train.T, 4)) - np.power(mse_train, 2)
        
    return kappa
```

### Return the p value for neural network
```
def get_dnn_p_val(mse_train, mse_test, n_train, n_test, kappa):
    test_stat = 1/np.sqrt(kappa*(1/n_train + 1/n_test))*(mse_train - mse_test)
    p_val = norm.sf(abs(test_stat)) * 2
    print(test_stat)
    print(p_val)
    return p_val
```

## Usage
```python
from sim_utils4 import *
from sim_utils5 import *
```