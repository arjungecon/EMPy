import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy.stats import norm 
import time 
import json
import torch 


def Expectation_Maximization_Tensorflow (data,alpha__0,mean_mu__0,var_v__0,k=2,treshold=False,nb_iter=100,eps = 10e-5):
    
    """
        Runs the Expectation-Maximization algorithm using Tensoflow
        :param data: Data used in the EM algorithm
        :param alpha__0: Initial parameter for weight
        :param mean_mu__0: Initial parameter for mean 
        :param var_v__0: Initial parameter for variance
        :param treshold: float to activate the loop of likelihood (default False)
        :param epsilon: Precision on the likelihood (default 1e-5)
        :param numb_iter: Number of iterations (default 100)
        :return: Estimated parameters ( weight, mean, standard deviation )
    """
    
    
    const_gauss = tf.constant(np.log(2 * np.pi) * 1 , dtype=tf.float64)


    # BUILD COMPUTATIONAL GRAPH

    # input of the model 
    
    input = tf.placeholder(tf.float64, [None, 1])

    # computing input statistics
    
    mean_0 = tf.reduce_mean(input, 0)
    diff_0 = tf.squared_difference(input, tf.expand_dims(mean_0, 0))
    var_0 = tf.reduce_sum(diff_0, 0) / tf.cast(tf.shape(input)[0], tf.float64)
    var_mean_0 = tf.cast(tf.reduce_sum(var_0) / k / 1 , tf.float64)

    # Initial values for the algorithm 
    
    mean_init = tf.placeholder_with_default( mean_mu__0,shape=[k, 1] )
    
    var_init = tf.placeholder_with_default( var_v__0,shape=[k, 1] )
    
    weight_init = tf.placeholder_with_default(alpha__0 , shape=[k] )
    

    # Variables for training 
    
    mean_mu = tf.Variable(mean_init, dtype=tf.float64)
    var_v = tf.Variable(var_init, dtype=tf.float64)
    weight_pi = tf.Variable(weight_init, dtype=tf.float64)

    # Step 1 : Expectation  ( logsumexp , normal )
    
    diff_squared = tf.squared_difference(tf.expand_dims(input, 0), tf.expand_dims(mean_mu, 1))
    diff_squared_var = tf.reduce_sum(diff_squared / tf.expand_dims(var_v, 1), 2)
    gauss_log = tf.expand_dims(const_gauss + tf.reduce_sum(tf.log(var_v), 1), 1)
    gauss_log_com = -1/2 * (gauss_log + diff_squared_var)
    weight_log = gauss_log_com + tf.expand_dims(tf.log(weight_pi), 1)
    logg = tf.expand_dims(tf.reduce_max(weight_log, 0), 0)
    exp_logg = tf.exp(weight_log - logg)
    exp_logg_sum = tf.reduce_sum(exp_logg, 0)
    gamma = exp_logg / exp_logg_sum

    # Step 2 : Maximization 
    
    gamma_sum = tf.reduce_sum(gamma, 1)
    gamma_weighted = gamma / tf.expand_dims(gamma_sum, 1)
    mean_estim = tf.reduce_sum(tf.expand_dims(input, 0) * tf.expand_dims(gamma_weighted, 2), 1)
    diff_estim = tf.squared_difference(tf.expand_dims(input, 0), tf.expand_dims(mean_estim, 1))
    var_estim = tf.reduce_sum(diff_estim * tf.expand_dims(gamma_weighted, 2), 1)
    weights_estim = gamma_sum / tf.cast(tf.shape(input)[0], tf.float64)
    var_estim *= tf.expand_dims(gamma_sum, 1)
    var_estim /= tf.expand_dims(gamma_sum, 1)
    
     
        
    # Compute Loglikelihood 
    
    log_likelihood = tf.reduce_sum(tf.log(exp_logg_sum)) + tf.reduce_sum(logg)
    mean_log_likelihood = log_likelihood / tf.cast(tf.shape(input)[0] * tf.shape(input)[1], tf.float64)

    # assignement of new values for parameters 
    
    train_step = tf.group( mean_mu.assign(mean_estim), var_v.assign(var_estim), weight_pi.assign(weights_estim) )

    # RUN COMPUTATIONAL GRAPH

    with tf.Session() as sess:
        
        # initializing trainable variables ( cout ++ ( creacte object ) )
        
        sess.run( tf.global_variables_initializer(), feed_dict={input: data, mean_init: data[:k],} )
        
        log_likelihood_init = -np.inf

        for step in range(nb_iter):
            
            # Training step execution 
            
            log_likelihood, _ = sess.run( [mean_log_likelihood, train_step], feed_dict={input: data} )
        

            
            if treshold==True: 
                
                if step > 0:

                    log_likelihood_diff = log_likelihood - log_likelihood_init
                    

                    if log_likelihood_diff <= eps:

                        break


                log_likelihood_init = log_likelihood


        # Get the final Values 
        
        means_em = mean_mu.eval(sess)
        variance_em = var_v.eval(sess)
        weight_em=weight_pi.eval(sess)
    
        
    return means_em,np.sqrt(variance_em),weight_em

