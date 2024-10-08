import tensorflow as tf
import numpy as np

xy = np.loadtxt('03train.txt', unpack=True,
                dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1,len(x_data)],-1,1))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(501):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(cost), sess.run(W))
