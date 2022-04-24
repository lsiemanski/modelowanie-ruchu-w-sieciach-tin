import numpy as np
import os

DATA_PATH = "DATA"
GENERATOR_NAME = "1"

data_path = os.path.join(DATA_PATH, GENERATOR_NAME)

if not os.path.exists(data_path):
    os.makedirs(data_path)

nodes = [(5, 8), (8, 5), (5, 12), (8, 12)]

files = os.listdir('traffic')
Y = np.zeros((len(nodes), len(files)))

predict_from = 5
predict_to = 8

for i, file in enumerate(files):
    for j, (predict_from, predict_to) in enumerate(nodes):
        file_data = np.loadtxt('traffic/' + file)
        Y[j, i] = file_data[predict_from, predict_to]


for i, (predict_from, predict_to) in enumerate(nodes):
    np.save(os.path.join(
        data_path, f"Y_{predict_from}_{predict_to}.npy"), Y[i])
