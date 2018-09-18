import numpy as np
import pandas as pd
import time
import tensorflow as tf
import cv2

Program_Start = time.time()
max_step = 3000
batch_size = 128
#
# data_test = np.load('data_test.npy')
# label_test = np.load('label_test.npy')
# data = np.load('data.npy')
# label = np.load('label.npy')
#
# print(data.shape)
# print(label.shape)
#
# print(data_test.shape)
# print(label_test.shape)
#
# batch1 = data[:batch_size, :, :, :]
# print(batch1.shape)

#  no dropout, conv-pool-conv-pool-fc1-fc2

#  data
x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x_input')
y_input = tf.placeholder(tf.float32, shape=[None, 10], name='y_input')

# CONV : -1x32x32x3 -> -1x32x32x32

kernel = tf.Variable(tf.truncated_normal([5, 5, 3, 32], dtype=tf.float32, stddev=1e-1, name='weights'))
conv = tf.nn.conv2d(x_input, kernel, [1, 1, 1, 1], padding='SAME')
print('conv.shape=', conv.shape)
bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32]), name='bias')
print('bias.shape=', bias.shape)
Wx_plus_b = tf.nn.bias_add(conv, bias)
print('bias.shpae=', Wx_plus_b.shape)
conv1 = tf.sigmoid(Wx_plus_b, name='conv1')  # or Use ReLu
print('conv1.shape=', conv1.shape)

pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1',
                       padding='VALID')  # or Max-pool
print('pool1.shape=', pool1.shape)
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
print('norm1.shape=', norm1.shape)

# CONV : -1x32x32x32 -> -1x32x32x64
kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], dtype=tf.float32, stddev=1e-1, name='weights'))
conv = tf.nn.conv2d(norm1, kernel, [1, 3, 3, 1], padding='SAME')
print('conv,shape=', conv.shape)
bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]), name='bias')
print('bias.shape=', bias.shape)
Wx_plus_b = tf.nn.bias_add(conv, bias)
print('bias.shpae=', Wx_plus_b.shape)
conv2 = tf.sigmoid(Wx_plus_b, name='conv2')
print('conv2.shape=', conv2.shape)

pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool2', padding='VALID')
print('pool2.shape=', pool2.shape)
norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
print('norm2.shape=', norm2.shape)
pass

# flatten
x_flatten = tf.layers.flatten(norm2)
print(x_flatten.shape)

W_fc1 = tf.Variable(tf.truncated_normal([1024, 256], dtype=tf.float32, mean=0, stddev=1e-1), name='W_fc1')
fc1 = tf.nn.sigmoid(tf.matmul(x_flatten, W_fc1))

W_fc2 = tf.Variable(tf.truncated_normal([256, 10], dtype=tf.float32, mean=0, stddev=1e-1), name='W_fc2')
fc2 = tf.nn.sigmoid(tf.matmul(fc1, W_fc2))
print('fc2.shape=', fc2.shape)

init = tf.global_variables_initializer()

# 定义训练的loss函数。
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_input))

# 定义优化器，学习率设置为0.09，学习率可以设置为其他的数值。
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.09).minimize(loss)

# 定义准确率
valaccuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, 1), tf.argmax(y_input, 1)), tf.float32))

# indices = np.arange(40000)
# np.random.shuffle(indices)
# a = data[0, :, :, :]
# print(a.shape)
# print(a)
# b = data[1, :, :, :][np.newaxis, :]
# c = data[2, :, :, :]
# anew = a[np.newaxis, :]
# bnew = b[np.newaxis, :]
# cnew = c[np.newaxis, :]
# d = anew
# d = np.vstack((d, b))
# d = np.vstack((d, cnew))
# print(d.shape)

# tmp_y = pd.get_dummies(label[100])
# print(tmp_y.shape)

# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(1):
#         indices = np.arange(40000)
#         np.random.shuffle(indices)
#         for i in range(0, 40000, 100):
#             x_in = []
#             y_in = []
#             for j in range(0, 100):
#                 tmp_x = data[indices[i + j]][np.newaxis, :]
#                 tmp_y = label[indices[i + j]]
#                 if j == 0:
#                     x_in = tmp_x
#                     y_in = tmp_y
#                 else:
#                     x_in = np.vstack((x_in, tmp_x))
#                     y_in = np.hstack((y_in, tmp_y))
#             y_in = pd.get_dummies(y_in)
#             a, b = sess.run([optimizer, loss], feed_dict={x_input: x_in, y_input: y_in})
#             print("\r%lf\n" % b, end='')
#
#         acc = 0
#         for i in range(0, 10000, 200):
#             x_in = data_test[i:i + 200, :, :, :]
#             y_in = pd.get_dummies(label_test[i:i + 200])
#             acc += sess.run(valaccuracy, feed_dict={x_input: x_in, y_input: y_in})
#         print("Epoch ", epoch, ': validation rate: ', acc / 50)

Program_End = time.time()
print('\n', Program_End - Program_Start)