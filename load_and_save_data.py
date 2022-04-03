import numpy as np
import os

n = 5

files = os.listdir('traffic')
X = np.zeros((len(files), 28*28-1))
Y = np.zeros((len(files), 1))

predict_from = 5
predict_to = 8

for i, file in enumerate(files):
    file_data = np.loadtxt('traffic/' + file)
    Y[i, :] = file_data[predict_from, predict_to]
    file_data = np.delete(file_data, predict_from*28+predict_to)
    X[i, :] = file_data.reshape(1, (28*28-1))

if n != 1:
    Y_new = np.zeros((Y.shape[0]-n, n))
    for i in range(Y.shape[0]-n):
        Y_new[i, :] = np.ravel(Y[i:i+n, :])
    Y = Y_new
    X = X[0:X.shape[0]-n, :]

np.save("X.npy", X)
np.save("Y.npy", Y)

