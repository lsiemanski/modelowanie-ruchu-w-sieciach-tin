import numpy as np
from matplotlib import pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


static_mape_errors = np.load("static_MAPE.npy")
dynamic_mape_errors = np.load("dynamic_MAPE.npy")

static_mape_errors = np.mean(static_mape_errors, axis=0)
dynamic_mape_errors = np.mean(dynamic_mape_errors, axis=0)

print(np.mean(static_mape_errors))
print(np.mean(dynamic_mape_errors))

static_mape_errors = moving_average(static_mape_errors, n=20)
dynamic_mape_errors = moving_average(dynamic_mape_errors, n=20)

x_axis = range(len(static_mape_errors))


plt.title("Compare MLP - without and with refitting")
plt.plot(x_axis, static_mape_errors, label='STATIC', color='r', linewidth=1)
plt.plot(x_axis, dynamic_mape_errors, label='DYNAMIC', color='b', linewidth=1)
plt.xlabel('Chunk nr')
plt.ylabel('MAPE')
plt.legend()
plt.show()
