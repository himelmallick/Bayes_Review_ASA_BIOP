import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Priors for regular models
def get_normal_prior(scale):
    return tfd.Normal(loc=0., scale=scale)

def get_laplace_prior(scale):
    return tfd.Laplace(loc=0., scale=scale)

def get_horseshoe_prior(scale):
    return tfd.Horseshoe(scale=scale)

# Horseshoe prior for tied model
class TiedHorseshoe:
    def __init__(self, input_dim):
        self.global_shrinkage = tfd.HalfCauchy(loc=0., scale=1.)
        self.local_shrinkage = tfd.HalfCauchy(loc=np.zeros((input_dim, 1)).astype(np.float32), scale=np.ones((input_dim, 1)).astype(np.float32))

    def log_prob(self, global_shrinkage_val, local_shrinkage_vals, weight_val):
        log_prob = self.global_shrinkage.log_prob(global_shrinkage_val)
        log_prob += tf.reduce_sum(self.local_shrinkage.log_prob(local_shrinkage_vals))
        weight = tfd.Normal(loc=0., scale=local_shrinkage_vals*global_shrinkage_val)
        log_prob += tf.reduce_sum(weight.log_prob(weight_val))
        return log_prob

    def sample(self, sample_shape):
        global_shrinkage_val = self.global_shrinkage.sample()
        local_shrinkage_vals = self.local_shrinkage.sample()
        weight = tfd.Normal(loc=0., scale=local_shrinkage_vals*global_shrinkage_val)
        return weight.sample(sample_shape)

# Horseshoe prior for tied model with fixed global shrinkage
class TiedHorseshoeGlobal:
    def __init__(self, global_shrinkage, input_dim):
        self.global_shrinkage = global_shrinkage
        self.local_shrinkage = tfd.HalfCauchy(loc=np.zeros((input_dim, 1)).astype(np.float32), scale=np.ones((input_dim, 1)).astype(np.float32))

    def log_prob(self, local_shrinkage_vals, weight_val):
        log_prob = tf.reduce_sum(self.local_shrinkage.log_prob(local_shrinkage_vals))
        weight = tfd.Normal(loc=0., scale=local_shrinkage_vals*self.global_shrinkage)
        log_prob += tf.reduce_sum(weight.log_prob(weight_val))
        return log_prob

    def sample(self, sample_shape):
        local_shrinkage_vals = self.local_shrinkage.sample()
        weight = tfd.Normal(loc=0., scale=local_shrinkage_vals*self.global_shrinkage)
        return weight.sample(sample_shape)