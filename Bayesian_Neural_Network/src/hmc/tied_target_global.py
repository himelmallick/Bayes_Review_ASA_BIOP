import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from src.hmc.priors import TiedHorseshoeGlobal
from src.hmc.target import Target

class TiedTargetGlobal(Target):
    def __init__(self, X, y, layers, prior, global_shrinkage):
        super().__init__(X, y, layers, prior)
        self.tied_horseshoe = TiedHorseshoeGlobal(global_shrinkage, input_dim=layers[0])

    def get_prior_log_prob_fn(self):
        def prior_log_prob(local_shrinkage_vals, Ws, bs):
            # First layer weights (without bias)
            log_prob = self.tied_horseshoe.log_prob(local_shrinkage_vals, Ws[0])

            # Remaining weights
            if len(Ws)>1:
                for W in Ws[1:]:
                    log_prob += tf.reduce_sum(self.prior.log_prob(W))
            for b in bs:
                log_prob += tf.reduce_sum(self.prior.log_prob(b))
            return log_prob
        return prior_log_prob

    def log_prob(self, weights, likelihood_log_prob_fn, prior_log_prob_fn):
        local_shrinkage_vals, Ws, bs = self.split_weights(weights)

        # Data likelihood
        log_prob = likelihood_log_prob_fn(Ws, bs)

        # Prior
        log_prob += prior_log_prob_fn(local_shrinkage_vals, Ws, bs)

        return log_prob

    def split_weights(self, weights):
        # Read horseshoe parameters
        input_dim = self.layers[0]
        local_shrinkage_vals = tf.math.softplus(weights[0:input_dim])
        local_shrinkage_vals = tf.reshape(local_shrinkage_vals, (input_dim, 1))

        num_used = input_dim

        # Read model weights
        Ws = []
        bs = []
        input_dim = self.layers[0]
        for output_dim in self.layers[1:]:
            W = tf.reshape(weights[num_used:num_used + input_dim*output_dim], [input_dim, output_dim])
            Ws.append(W)
            num_used += input_dim*output_dim
            
            b = tf.reshape(weights[num_used:num_used + output_dim], [output_dim])
            bs.append(b)
            num_used += output_dim
            
            input_dim = output_dim
        return local_shrinkage_vals, Ws, bs

    def get_fixed_shrinkage_init_state(self, num_chains):
        input_dim = self.layers[0]
        shrinkage_init_state = tf.ones((input_dim)) * tf.math.log(tf.math.exp(1.) - 1.)

        num_weights = 0
        for output_dim in self.layers[1:]:
            num_weights += input_dim*output_dim + output_dim
            input_dim = output_dim   
        
        init_state = tf.TensorArray(tf.float32, size=num_chains)
        
        for i in range(num_chains):
            weights_init_state_i = self.prior.sample([num_weights])
            init_state_i = tf.concat([shrinkage_init_state, weights_init_state_i], axis=0)
            init_state = init_state.write(i, init_state_i)
        
        return init_state.stack()

    def model_mean(self, weights, X):
        _, Ws, bs = self.split_weights(weights)
        A = X
        for W, b in zip(Ws, bs):
            z = tf.tensordot(A, W, axes=[[1], [0]]) + b
            A = tf.nn.sigmoid(z)
        probs = A
        if probs.shape[1]==1:
            probs = tf.squeeze(probs, axis=[1])
        return probs