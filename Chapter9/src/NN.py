import tensorflow as tf
import numpy as np

xy = np.loadtxt('07train.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

print x_data
print y_data

X = tf.placeholder(tf.float32, name='x-input')
Y = tf.placeholder(tf.float32, name='y-input')

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='weight1')
W2 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight2')

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(20000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w1), sess.run(w2)

    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction], feed_dict={X: x_data, Y: y_data})
    print "accuracy", accuracy.eval({X: x_data, Y: y_data})
