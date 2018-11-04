import numpy as np
import tensorflow as tf
import time
import pandas as pd


x = np.load('data.npy')  # 480x340x16
y = np.load('label.npy')  # 480x10

indices = np.arange(480)
np.random.shuffle(indices)

x_in = x[indices[0]]
y_in = y[indices[0]]
for j in range(1, 480):
    x_in = np.vstack((x_in, x[indices[j]]))
    y_in = np.hstack((y_in, y[indices[j]]))
x_in = x_in.reshape([480, 340, 16, 1])
a = y_in.reshape([480, 10])
y_in = pd.get_dummies(y_in)
print(y_in.shape)

print(1)
