import numpy as np
import tensorflow as tf
import time
import pandas as pd


x_in = np.load('data.npy')  # 480x340x16
y_in = np.load('label.npy')  # 480x10

# a = np.arange(0, 12, dtype=np.float32)
# np.random.shuffle(a)
# print(a)
# d = pd.get_dummies(a)
# print(d)


start_time = time.time()

# input_shape
with tf.name_scope('Input'):
    x_input = tf.placeholder(tf.float32, shape=[None, 340, 16, 1], name='x_input')
    y_input = tf.placeholder(tf.float32, shape=[None, 10], name='y_input')
    pass

with tf.name_scope('Conv1'):
    W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=1e-1, name='weights'))
    W1x = tf.nn.conv2d(x_input, W1, [1, 1, 1, 1], padding='SAME')
    b1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[16]), name='bias')
    Wx_plus_b = tf.nn.bias_add(W1x, b1)
    conv1 = tf.sigmoid(Wx_plus_b, name='conv1')  # or Use ReLu
    print('W1x.shape=', W1x.shape)
    print('b1.shape=', b1.shape)
    print('W1x+b.shpae=', Wx_plus_b.shape)
    print('conv1_out.shape=', conv1.shape)

    pool1 = tf.nn.avg_pool(conv1, ksize=[1, 11, 1, 1], strides=[1, 11, 1, 1], name='pool1',
                           padding='VALID')  # or Max-pool
    print('pool1_out.shape=', pool1.shape)

    # Local Response Normalization -> generalization
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print('norm1.shape=', norm1.shape)
    pass

with tf.name_scope('Conv2'):
    W2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=1e-1, name='weights'))
    W2x = tf.nn.conv2d(norm1, W2, [1, 1, 1, 1], padding='SAME')
    b2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32]), name='bias')
    W2x_plus_b = tf.nn.bias_add(W2x, b2)
    conv2 = tf.sigmoid(W2x_plus_b, name='conv1')  # or Use ReLu
    print('W2x.shape=', W2x.shape)
    print('b2.shape=', b2.shape)
    print('W2x+b.shpae=', W2x_plus_b.shape)
    print('conv2_out.shape=', conv2.shape)

    pool2 = tf.nn.avg_pool(conv2, ksize=[1, 5, 1, 1], strides=[1, 3, 1, 1], name='pool1',
                           padding='VALID')  # or Max-pool
    print('pool2_out.shape=', pool2.shape)

    # Local Response Normalization -> generalization
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print('norm2.shape=', norm2.shape)
    pass

# flatten
x_flatten = tf.layers.flatten(norm2)
print('x_flatten.shape=', x_flatten.shape)

#
W_fc1 = tf.Variable(tf.truncated_normal([4068, 1024], dtype=tf.float32, mean=0, stddev=1e-1), name='W_fc1')
fc1 = tf.nn.sigmoid(tf.matmul(x_flatten, W_fc1))

W_fc2 = tf.Variable(tf.truncated_normal([1024, 256], dtype=tf.float32, mean=0, stddev=1e-1), name='W_fc2')
fc2 = tf.nn.sigmoid(tf.matmul(fc1, W_fc2))

W_fc3 = tf.Variable(tf.truncated_normal([256, 10], dtype=tf.float32, mean=0, stddev=1e-1), name='W_fc2')
fc3 = tf.nn.sigmoid(tf.matmul(fc1, W_fc3))
print('fc_out.shape=', fc3.shape)

#
y_predict = fc3

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in, logits=y_predict))
with tf.name_scope('Train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_in, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1):
        indices = np.arange(480)
        np.random.shuffle(indices)
        for i in range(0, 480, 20):
            x_in = []
            y_in = []
            for j in range(0, 100):
                tmp_x = x_in[indices[i + j]][np.newaxis, :]
                tmp_y = y_in[indices[i + j]]
                if j == 0:
                    x_in = tmp_x
                    y_in = tmp_y
                else:
                    x_in = np.vstack((x_in, tmp_x))
                    y_in = np.hstack((y_in, tmp_y))
            y_in = pd.get_dummies(y_in)
            a, b = sess.run([optimizer, loss], feed_dict={x_input: x_in, y_input: y_in})
            print("\r%lf\n" % b, end='')

        acc = 0
        for i in range(0, 10000, 200):
            x_in = data_test[i:i + 200, :, :, :]
            y_in = pd.get_dummies(label_test[i:i + 200])
            acc += sess.run(valaccuracy, feed_dict={x_input: x_in, y_input: y_in})
        print("Epoch ", epoch, ': validation rate: ', acc / 50)


end_time = time.time()
print(1)


