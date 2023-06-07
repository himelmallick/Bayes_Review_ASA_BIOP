import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Target:
    def __init__(self, X, y, layers, prior):
        self.X = X
        self.y = y
        self.layers = layers
        self.prior = prior

    def get_target_log_prob_fn(self):
        # Class weights
        pos = np.sum(self.y)
        neg = len(self.y) - pos
        total = pos + neg

        weight0 = (1 / neg)*(total)/2.0
        weight1 = (1 / pos)*(total)/2.0

        y_weights = [weight0 if val==0 else weight1 for val in self.y]
        y_weights = tf.constant(y_weights, dtype=tf.float32)

        likelihood_log_prob_fn = self.get_likelihood_log_prob_fn(y_weights)
        prior_log_prob_fn = self.get_prior_log_prob_fn()

        args = (likelihood_log_prob_fn, prior_log_prob_fn)

        target_fn = lambda weights: self.target_log_prob(weights, args)
        
        return target_fn

    def get_likelihood_log_prob_fn(self, y_weights):
        def likelihood_log_prob(Ws, bs):
            likelihood = self.model(Ws, bs)
            log_prob = tf.reduce_sum(y_weights*likelihood.log_prob(self.y))
            # log_prob = tf.reduce_sum(likelihood.log_prob(self.y))
            return log_prob
        return likelihood_log_prob

    def get_prior_log_prob_fn(self):
        def prior_log_prob(Ws, bs):
            log_prob = 0
            for W in Ws:
                log_prob += tf.reduce_sum(self.prior.log_prob(W))
            for b in bs:
                log_prob += tf.reduce_sum(self.prior.log_prob(b))
            return log_prob
        return prior_log_prob

    def target_log_prob(self, weights, args):
        """Target log-probability as a function of states."""
        log_prob_weights_fn = lambda weights: self.log_prob(weights, *args)
        return tf.vectorized_map(log_prob_weights_fn, weights)

    def log_prob(self, weights, likelihood_log_prob_fn, prior_log_prob_fn):
        Ws, bs = self.split_weights(weights)

        # Data likelihood
        log_prob = likelihood_log_prob_fn(Ws, bs)

        # Prior
        log_prob += prior_log_prob_fn(Ws, bs)

        return log_prob
    
    def split_weights(self, weights):
        Ws = []
        bs = []
        num_used = 0
        input_dim = self.layers[0]
        for output_dim in self.layers[1:]:
            W = tf.reshape(weights[num_used:num_used + input_dim*output_dim], [input_dim, output_dim])
            Ws.append(W)
            num_used += input_dim*output_dim
            
            b = tf.reshape(weights[num_used:num_used + output_dim], [output_dim])
            bs.append(b)
            num_used += output_dim
            
            input_dim = output_dim
        return Ws, bs

    def get_init_state(self, num_chains):
        num_weights = 0
        input_dim = self.layers[0]
        for output_dim in self.layers[1:]:
            num_weights += input_dim*output_dim + output_dim
            input_dim = output_dim
        init_state = self.prior.sample([num_chains, num_weights])
        return init_state

    def model(self, Ws, bs):
        A = self.X
        for W, b in zip(Ws[:-1], bs[:-1]):
            Z = tf.tensordot(A, W, axes=[[1], [0]]) + b
            A = tf.nn.sigmoid(Z)
        logits = tf.tensordot(A, Ws[-1], axes=[[1], [0]]) + bs[-1]
        if logits.shape[1]==1:    
            logits = tf.squeeze(logits, axis=[1])
        y = tfd.Bernoulli(logits=logits)
        return y

    def model_mean(self, weights, X):
        Ws, bs = self.split_weights(weights)
        A = X
        for W, b in zip(Ws, bs):
            z = tf.tensordot(A, W, axes=[[1], [0]]) + b
            A = tf.nn.sigmoid(z)
        probs = A
        if probs.shape[1]==1:
            probs = tf.squeeze(probs, axis=[1])
        return probs