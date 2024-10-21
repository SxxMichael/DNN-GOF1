from sim_utils6 import *

p_val_lm, p_val_nn_hommel, p_val_nn_cauchy, p_val_dnn_hommel, p_val_dnn_cauchy, p_val_dnn2_hommel, p_val_dnn2_cauchy = main(n_sample = np.array([200, 500, 1000, 2000]), tag = 3, nS = 2, gamma = 0.5)

res = np.hstack((p_val_lm, p_val_nn_hommel, p_val_nn_cauchy, p_val_dnn_hommel, p_val_dnn_cauchy, p_val_dnn2_hommel, p_val_dnn2_cauchy))
print_array(res)