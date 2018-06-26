# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:38:36 2018

@author: Łukasz Treszczotko
"""

# beta version

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from fbm import FBM
import time


print(80*'_')

num_epochs = 10
#series_length = 999
n_samples = 1000
batch_size = 10
true_H = 0.78
train_sample_length = 100
n = 10
sample_length = train_sample_length + n
state_size = 10
learning_rate = 0.02
test_size = 100

print('Predicting fBm using LSTM: %d-STEPS AHEAD.\n' % (n))





def generate_data(n_samples, sample_length, H):
    """ Generates sample fBm paths using the fbm module."""
    data = np.array([FBM(n=sample_length, hurst=H, length=1,
                         method='daviesharte').fbm() for j in range(n_samples)])
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    
    return shuffled_data
    
def LSTM_cell(num_units, output_size):
  return tf.contrib.rnn.BasicLSTMCell(
      num_units=num_units)
      
     # reuse=tf.get_variable_scope().reuse

def GRU_cell(num_units, output_size):
  return tf.contrib.rnn.GRUCell(num_units=num_units)
  

  
tf.reset_default_graph()
print('Assembling the graph... ')
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, sample_length+1, 1], name='X')
    #Y = tf.placeholder(tf.float32, [None, n, 1], name='Y')
    #init_state = tf.placeholder(tf.float32, [batch_size, state_size], name='init')
    
with tf.name_scope('LSTM_cell'):
    with tf.variable_scope('LSTM'):
        cell_LSTM = tf.contrib.rnn.OutputProjectionWrapper(
            LSTM_cell(num_units=state_size, output_size=n), output_size=n, reuse=False)
        
        outputs_LSTM, states_LSTM = tf.nn.dynamic_rnn(cell_LSTM, X, dtype=tf.float32)
        
with tf.name_scope('GRU_cell'):
    with tf.variable_scope('GRU'):
        cell_GRU = tf.contrib.rnn.OutputProjectionWrapper(
            GRU_cell(num_units=state_size, output_size=n), output_size=n)
        outputs_GRU, states_GRU = tf.nn.dynamic_rnn(cell_GRU, X, dtype=tf.float32)
    
    
    
    
with tf.name_scope('training_ops'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss_counter_1 = tf.constant(0, dtype=tf.float32)
    for j in range(sample_length - n):
        loss_counter_1 += tf.reduce_mean(tf.square(outputs_LSTM[:,j,:]-X[:, j :j + n :, 0]))
        #loss = tf.reduce_mean(tf.reduce_mean(tf.square(outputs[:,-1,:]-Y[:,-n:,0]), axis=0), axis=0)
    loss_LSTM  = loss_counter_1
            
    loss_counter_2 = tf.constant(0, dtype=tf.float32)
    for j in range(sample_length - n):
        loss_counter_2 += tf.reduce_mean(tf.square(outputs_GRU[:,j,:]-X[:, j :j + n :, 0]))
        #loss = tf.reduce_mean(tf.reduce_mean(tf.square(outputs[:,-1,:]-Y[:,-n:,0]), axis=0), axis=0)
    loss_GRU  = loss_counter_2
    
    training_op_1 = optimizer.minimize(loss_LSTM)
    training_op_2 = optimizer.minimize(loss_GRU)
    
    
init = tf.global_variables_initializer()


n_batches = n_samples // batch_size
print('Generating samples... ')
data = generate_data(n_samples=n_samples, sample_length=sample_length, H=true_H)
data = np.expand_dims(data, axis=2)

def train_L_G_T(num_epochs=num_epochs, H=true_H, state_size=state_size, 
                learning_rate=learning_rate):
    """ Trains the the LSTM and GRU recurrent neural networks for predicting 
        fractional Brownian motion. Saves the trained model for later use."""
    

    
    print('Training the network... ')
    print()
    with tf.Session() as sess:
        saver = tf.train.Saver()    
        start_global_time = time.time()
        sess.run(init)
        writer = tf.summary.FileWriter('graphs/beta_version', sess.graph)
        for epoch in range(num_epochs):
            for step in range(n_batches):
                X_batch = data[step*batch_size: step*batch_size + batch_size, :, :]
                _ , __, output_LSTM, output_GRU, current_loss_LSTM, current_loss_GRU = sess.run(
                    [training_op_1, training_op_2,  outputs_LSTM,  outputs_GRU, 
                     loss_LSTM, loss_GRU], feed_dict={X: X_batch})
            
            if (epoch + 1) % 2 == 0:
                time_now = time.time()
                time_per_epoch = (time_now-start_global_time)/(epoch+1)
                MSE_LSTM = loss_LSTM.eval(feed_dict={X: X_batch})
                MSE_GRU = loss_GRU.eval(feed_dict={X: X_batch})
                print('Epoch nr: ', epoch+1, ' MSE_LSTM: ', MSE_LSTM, ' MSE_GRU: ', MSE_GRU)
                print('MSE_LSTM per step: ', MSE_LSTM / n, ' MSE_GRU per step: ', MSE_GRU / n)
                print('Expected time remaining: %.2f seconds.' % (time_per_epoch * (num_epochs - epoch)))
                print(80*'_')
            
        saver.save(sess, "./beta_version")
        writer.close()
        
    


train_L_G_T()
        
            


def test_MSE(test_size=test_size):
    print('Generating data fot testing... ')
    
    sample_in = generate_data(n_samples=test_size, sample_length=sample_length, H=true_H)
    sample = np.expand_dims(sample_in, axis=2)
    print('Evaluating losses... ')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./beta_version")
        
        with tf.variable_scope('LSTM'):
            temp_LSTM = sess.run(outputs_LSTM, feed_dict={X: sample})
        loss_counter_1 = 0
        for j in range(n):
            loss_counter_1 += np.mean(np.square(temp_LSTM[:,-n,j]-sample[:, -n + j, 0]), axis=0)
        MSE_LSTM = loss_counter_1
        
        temp_GRU = sess.run(outputs_GRU, feed_dict={X: sample})
        loss_counter_2 = 0
        for j in range(n):
            loss_counter_2 += np.mean(np.square(temp_GRU[:,-n,j]-sample[:, -n + j, 0]), axis=0)
        MSE_GRU = loss_counter_2
        benchmark_samples = np.zeros_like(sample_in)
        
        MSE_benchmark = 0
        for j in range(test_size):
            benchmark_samples[j] = sample_to_function(sample_in[j])
        for j in range(n):
            MSE_benchmark += np.mean(np.square(benchmark_samples[:, -n + j] -sample_in[:, -n + j]), axis=0)
        
            
    return n * MSE_LSTM, n * MSE_GRU, n * MSE_benchmark
    


def plot_test():
    
    sample = generate_data(n_samples=1, sample_length=sample_length, H=true_H)
    sample = np.expand_dims(sample, axis=2)
    sample_vector = sample[0,:,0]
    
    with tf.Session() as sess:
        
        saver = tf.train.Saver()  
        saver.restore(sess, "./beta_version")
        
        
        
        temp_LSTM = sess.run(outputs_LSTM, feed_dict={X: sample})
        predicted_values_LSTM = temp_LSTM[0,- n ,:]
            
        
        
        temp_GRU = sess.run(outputs_GRU, feed_dict={X: sample})
        predicted_values_LSTM = temp_GRU[0,- n ,:]
        
    sample_preds_LSTM = np.copy(sample_vector)
    sample_preds_LSTM[-n:] = predicted_values_LSTM
    
    sample_preds_GRU = np.copy(sample_vector)
    sample_preds_GRU[-n:] = predicted_values_LSTM
    
    
    plt.figure(figsize=(14,8))
    plt.plot(range(len(sample_vector)), sample_vector, 'b', linewidth=1, label='OBSERVED')
    plt.plot(range(len(sample_vector)), sample_preds_LSTM, 'r', label='LSTM', linewidth=2)
    plt.plot(range(len(sample_vector)), sample_preds_GRU, 'y', label='GRU', linewidth=2)
    plt.plot(range(len(sample_vector)), sample_to_function(sample_vector), 'g', label='THEORY', linewidth=2)
    plt.xlim([0,sample_length])
    plt.legend(loc=0)
    #plt.title('Error = %.5f' % (predicted_values - sample_vector[-1]))
    plt.show()

from scipy.integrate import quad

def psi(a, u, t, H):
    kappa = H - 1./2
    def integrand(z, a, u, kappa):
        return ((z ** kappa) * ((z - a) ** kappa))/(z - u)
    I = quad(integrand, a, t, args=(a, u, kappa))
    return I[0] * (np.sin(np.pi * kappa) * (u ** (-kappa)) * ((a - u) ** (-kappa))) / np.pi
    

def sample_to_function(sample):
    #sample = np.squeeze(sample_input, axis=0)
    sample_known = sample[:train_sample_length+1]
    a = train_sample_length/(train_sample_length + n)
    step = 1./(train_sample_length + n)
    predicted_sample = np.zeros_like(sample)
    predicted_sample[:train_sample_length + 1] = sample_known
    for j in range(1, n+1):
        
        r = 0.
        for i in range(1, train_sample_length -1):
            r += psi(a, float(i) * step, a + j * step, true_H) * (sample[i + 1] - sample[i])
            
            # calculate the integral
        r += sample[train_sample_length]
            # add the last known value
        predicted_sample[train_sample_length + j] = r
    return predicted_sample
    
def LTB():
    import os
    os.system('tensorboard --logdir=' + 'C:/Users/user/Desktop/PYTHON/FBM_LSTM/graphs/')
    return
    
# variables_names = [v.name for v in tf.trainable_variables()]