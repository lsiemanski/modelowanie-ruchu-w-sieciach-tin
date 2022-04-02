import numpy as np
import os

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

np.save("X.npy", X)
np.save("Y.npy", Y)

