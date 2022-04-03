from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

REPEAT = 10

CHUNK_SIZE = 1000

X = np.load("X.npy")
Y = np.load("Y.npy")

failure_time = 16000

X_train = X[:failure_time, :]
X_test = X[failure_time:, :]
Y_train = Y[:failure_time, :]
Y_test = Y[failure_time:, :]

# Y_train = np.ravel(Y_train)
# Y_test = np.ravel(Y_test)

chunks_number = int(X_test.shape[0] / 100)

X_test_chunks = np.array(np.array_split(X_test, chunks_number))
Y_test_chunks = np.array(np.array_split(Y_test, chunks_number))


static_mape_errors = np.zeros((REPEAT, chunks_number))
dynamic_mape_errors = np.zeros((REPEAT, chunks_number))

for i in range(REPEAT):
    # without re-fit
    model = Sequential()
    model.add(Dense(100))
    model.add(Dense(Y_test.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train)

    for c in range(chunks_number):
        prediction = model.predict(X_test_chunks[c])
        static_mape_errors[i, c] = mean_absolute_percentage_error(
            Y_test_chunks[c], prediction)

    # with re-fit
    model = Sequential()
    model.add(Dense(100))
    model.add(Dense(Y_test.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train)

    for c in range(chunks_number):
        prediction = model.predict(X_test_chunks[c])
        dynamic_mape_errors[i, c] = mean_absolute_percentage_error(
            Y_test_chunks[c], prediction)

        model.fit(X_test_chunks[c], Y_test_chunks[c])

    print('RUN: ', i + 1, ' - static mean: ', static_mape_errors[i].mean())
    print('RUN: ', i + 1, ' - dynamic mean: ', dynamic_mape_errors[i].mean())

np.save("static_MAPE.npy", static_mape_errors)
np.save("dynamic_MAPE.npy", dynamic_mape_errors)
