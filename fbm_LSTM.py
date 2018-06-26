# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:21:09 2018

@author: user
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fbm import FBM

print(80*'_')
print('Predicting fBm using LSTM: ONE STEP AHEAD.\n')

num_epochs = 50
#series_length = 999
n_samples = 100
batch_size = 20
true_H = 0.25
sample_length = 100
state_size = 4
learning_rate = 0.01


def generate_data(n_samples, sample_length, H):
    
    return np.array([FBM(n=sample_length, hurst=H, length=1, method='daviesharte').fbm() for j in range(n_samples)])
    

tf.reset_default_graph()


with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, sample_length - 1, 1], name='X')
    Y = tf.placeholder(tf.float32, [None, 1, 1], name='Y')
    init_state = tf.placeholder(tf.float32, [batch_size, state_size], name='init')
    
with tf.name_scope('LTSM_cell'):
    cell_state = tf.placeholder(tf.float32, [batch_size, state_size], name='cell_state')
    hidden_state = tf.placeholder(tf.float32, [batch_size, state_size], name='hidden_state')
    init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
    cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=state_size), output_size=1)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    
with tf.name_scope('training_ops'):
    loss = tf.reduce_mean(tf.square(outputs[:,-1,0]-Y[:,0,0]), axis=0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
    
init = tf.global_variables_initializer()


n_batches = n_samples // batch_size

data = generate_data(n_samples=n_samples, sample_length=sample_length, H=true_H)
data = np.expand_dims(data, axis=2)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        for step in range(n_batches):
            X_batch, Y_batch = data[step: step + batch_size, : sample_length - 1], data[step: step + batch_size, -1:]
            _ , output, current_loss = sess.run([training_op, outputs, loss], feed_dict={X: X_batch, Y:Y_batch})
            
        if (epoch + 1) % 5 == 0:
            print('Epoch nr: ', epoch+1, ' MSE: ', loss.eval(feed_dict={X: X_batch, Y:Y_batch}))
            
    saver.save(sess, "./fbm_LSTM_model")
        
            
def predict_test():
    sample = generate_data(n_samples=1, sample_length=sample_length, H=true_H)
    sample = np.expand_dims(sample, axis=2)
    with tf.Session() as sess:
        saver.restore(sess, "./fbm_LSTM_model")
        temp = sess.run(outputs, feed_dict={X: sample[:,:sample_length - 1,:], Y: sample[:,-1:,:]})
        predicted_value = temp[0,-1,0]
        true_value = sample[0,-1,0]
    return predicted_value, true_value

def test_MSE(test_size=50):
    sample = generate_data(n_samples=50, sample_length=sample_length, H=true_H)
    sample = np.expand_dims(sample, axis=2)
    with tf.Session() as sess:
        saver.restore(sess, "./fbm_LSTM_model")
        temp = sess.run(outputs, feed_dict={X: sample[:,:sample_length - 1,:], Y: sample[:,-1:,:]})
        predicted_value = temp[:,-1,0]
        true_value = sample[:,-1,0]
        MSE = np.mean(np.square(predicted_value - true_value), axis=0)
    return MSE

def plot_test():
    sample = generate_data(n_samples=1, sample_length=sample_length, H=true_H)
    sample = np.expand_dims(sample, axis=2)
    sample_vector = sample[0,:,0]
    with tf.Session() as sess:
        saver.restore(sess, "./fbm_LSTM_model")
        temp = sess.run(outputs, feed_dict={X: sample[:,:sample_length - 1,:], Y: sample[:,-1:,:]})
        predicted_value = temp[0,-1,0]
    sample_pred = np.copy(sample_vector)
    sample_pred[-1] = predicted_value
    plt.figure(figsize=(12,6))
    plt.plot(range(len(sample_vector)), sample_vector, 'r', linewidth=2)
    plt.plot(range(len(sample_vector)), sample_pred, 'b')
    plt.title('Error = %.5f' % (predicted_value - sample_vector[-1]))
    plt.show()
    
    
    

    