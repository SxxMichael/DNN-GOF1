import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras import Model
from keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from math import pi
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.regularizers import L1
from keras import layers
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import cauchy

#-------------------------------------------------------------------------------------------------------------------#
def generate_data(nN = 100, n_x_true = 1, n_y_true = 1, low = -1, high = 1):
    X_mat = np.random.uniform(low = low, high = high, size = nN*(n_x_true+1)).reshape((n_x_true+1, nN))
    Y_true = np.power(X_mat[1,:], 2)
    return X_mat, Y_true.reshape((1, nN))
    
#-------------------------------------------------------------------------------------------------------------------#
def get_linear_reg_p_val(X, Y, beta0 = 0, tail = "two"):
    n,p = X.shape
    df = n-p-1
    X = np.hstack((np.ones((n,1)), X))
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, Y)
    XtX_inv = np.linalg.inv(XtX)
    C = np.diagonal(XtX_inv).reshape((p+1,1))
    beta_hat = np.dot(XtX_inv, Xty)
    Y_hat = np.dot(X, beta_hat)
    MSE = np.sum(np.power(Y-Y_hat, 2))/df
    t_score = (beta_hat - beta0)/np.sqrt(MSE * C)
    if tail == "two":
        p_value = t.sf(np.abs(t_score), df) * 2  # for two-tailed test
    elif tail == "left":
        p_value = t.sf(t_score, df)  # for left-tailed test
    elif tail == "right":
        p_value = t.sf(-t_score, df)  # for right-tailed test
    else:
        raise ValueError(
            "Invalid tail argument. Use 'two', 'left', or 'right'.")
    return p_value.reshape(p+1,)

#-------------------------------------------------------------------------------------------------------------------#
def scheduler(epoch, lr):
     if epoch < 50:
        return 0.1
     else:
        return 0.1 / np.log(np.exp(1)+epoch)
#-------------------------------------------------------------------------------------------------------------------#
def calculate_kappa(Y_train, predict_train,
                    plus = False):
    if plus:
        kappa = np.mean(np.power(Y_train - predict_train.T, 4))
    else:
        mse_train = np.mean(np.power(Y_train - predict_train.T, 2))
        kappa = np.mean(np.power(Y_train - predict_train.T, 4)) - np.power(mse_train, 2)
        
    return kappa
#-------------------------------------------------------------------------------------------------------------------#
#######################################
# Fit a shallow relu network
# training data
# X is a dxn matrix with n being the training sample size
# Y is a 1xn vector of responses
#
# n_h is the number of hidden units
#######################################
def fit_shallow_relu_nn(X, Y, n_h, 
                       optimizer = 'adam', epochs = 1000, batch_size = 1,
                       early_stopping = True,
                       validation_split = 0.2, patience = 5, min_delta = 1e-4,
                       verbose = 0, drop_rate = 0.2):
    d_train,n = X.shape

    callback_train = EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta,
                                   restore_best_weights=True)
    callback_lr = LearningRateScheduler(scheduler)
    model_ReLU_nn = keras.Sequential()
    model_ReLU_nn.add(layers.InputLayer(input_shape=(d_train,)))
    model_ReLU_nn.add(layers.Dense(n_h, activation = "relu"))
    model_ReLU_nn.add(layers.Dropout(drop_rate))
    model_ReLU_nn.add(layers.Dense(1))

    model_ReLU_nn.compile(optimizer=optimizer, loss='mse')

    if early_stopping:
        model_ReLU_nn.fit(X.T, Y.T, batch_size=batch_size, epochs=epochs, 
                          validation_split = validation_split, callbacks = [callback_train, callback_lr],
                          verbose = verbose)
    else:
        model_ReLU_nn.fit(X.T, Y.T, batch_size=batch_size, epochs=epochs, verbose = verbose,
                          callbacks = [callback_lr])
    
    predict_train = model_ReLU_nn.predict(X.T, verbose = verbose)
    mse_train = np.mean(np.power(Y - predict_train.T, 2))

    return mse_train, predict_train
    
#-------------------------------------------------------------------------------------------------------------------#
#######################################
# Fit a deep relu network
# training data
# X_train is a dxn matrix with n being the training sample size
# Y_train is a 1xn vector of responses
#
# testing data
# X_test is a dxm matrix with m being the testing sample size
# Y_test is a 1xm vector of responses
#
# n_h is a vector containing the number of hidden units in each layer
# num_layers is the number of layers in the deep network
#######################################
def fit_deep_relu_nn(X_train, Y_train, n_h, num_layers, 
                     optimizer = 'adam', epochs = 1000, batch_size = 1,
                     early_stopping = True,
                     patience = 5, validation_split = 0.2, min_delta = 1e-4,
                     verbose = 0, drop_rate = 0.2):
    d_train,n = X_train.shape
    callback_train = EarlyStopping(monitor='val_loss', 
                                   patience=patience,
                                   min_delta=min_delta,
                                   restore_best_weights = True)
    callback_lr = LearningRateScheduler(scheduler)
        
    input = Input(shape=(d_train,))
    inp = input
    for i in range(num_layers):
        x = Dense(n_h[i], activation = 'relu')(inp)
        x = layers.Dropout(drop_rate)(x)
        inp = x
    output = Dense(1)(x)
    model_ReLU_dnn = Model(input, output)

    model_ReLU_dnn.compile(optimizer = optimizer, loss = 'mse')

    if early_stopping:
        model_ReLU_dnn.fit(X_train.T, Y_train.T, batch_size = batch_size, epochs = epochs,
                          validation_split = validation_split, callbacks = [callback_train, callback_lr],
                          verbose = verbose)
    else:
        model_ReLU_dnn.fit(X_train.T, Y_train.T, batch_size = batch_size, epochs = epochs,
                           verbose = verbose, callbacks = [callback_lr])
        
    predict_train = model_ReLU_dnn.predict(X_train.T, verbose = verbose)
    mse_train = np.mean(np.power(Y_train - predict_train.T, 2))
    
    return mse_train, predict_train
    
#-------------------------------------------------------------------------------------------------------------------#
#############################
# This function is used to find the p-value for the DNN-GOF test
# mse_train, mse_test are the MSEs for the training data and testing data
# n_train, n_test are the sample sizes of the training and testing data
# kappa is the estimated 4th moment of random error term
#############################
def get_dnn_p_val(mse_train, mse_test, n_train, n_test, kappa):
    test_stat = 1/np.sqrt(kappa*(1/n_train + 1/n_test))*(mse_train - mse_test)
    p_val = norm.sf(abs(test_stat)) * 2
    print(test_stat)
    print(p_val)
    return p_val

#-------------------------------------------------------------------------------------------------------------------#
###############################
# p-value combination
###############################
def p_val_combine(p_val, method='cauchy'):
    if method == 'hommel':
        U = len(p_val)
        q = np.arange(U) + 1
        p_order = np.sort(p_val)
        C_U = np.sum(1/q)
        p_val_combine = np.minimum(1, np.min(C_U * (U/q) * p_order))
    elif method == 'cauchy':
        T = np.mean(np.tan((0.5 - p_val) * pi))
        p_val_combine = cauchy.sf(T) 
    else:
        raise ValueError("Invalid Method!")
    return p_val_combine

#-------------------------------------------------------------------------------------------------------------------#
def print_array(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for elem in a:
            print("{}".format(elem).rjust(3), end="\t")
        print(end="\n")
        
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
