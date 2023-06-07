import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

import src.hmc.target as target

@tf.function
def get_posterior(X, states, target):
    post_samples = get_posterior_samples(X, states, target)
    post_means = tf.math.reduce_mean(post_samples, axis=0)
    return post_samples, post_means

def get_posterior_samples(X, states, target):
    num_samples = states.shape[0]*states.shape[1]
    num_weights = states.shape[2]
    weights_samples = tf.reshape(states, (num_samples, num_weights))
    model_weights_fn = lambda weights: target.model_mean(weights, X)
    return tf.map_fn(model_weights_fn, weights_samples)

def eval_hmc(labels, predictions, text=None):
    if text is not None:
        print(text)
    # Accuracy
    acc = keras.metrics.BinaryAccuracy()
    acc.update_state(labels, predictions)
    print("Accuracy: {0:.3f}".format(acc.result().numpy()))
    
    # Balanced accuracy
    ba = sklearn.metrics.balanced_accuracy_score(labels, predictions > 0.5)
    print("Balanced accuracy: {0:.3f}".format(ba))

    # AUC
    auc = keras.metrics.AUC()
    auc.update_state(labels, predictions)
    print("AUC: {0:.3f}".format(auc.result().numpy()))
    
    # Binary cross entropy
    bce = keras.metrics.BinaryCrossentropy()
    bce.update_state(labels, predictions)
    print("Binary cross entropy: {0:.3f}".format(bce.result().numpy()))
    
    # Precision
    pre = keras.metrics.Precision()
    pre.update_state(labels, predictions)
    print("Precision: {0:.3f}".format(pre.result().numpy()))
    
    # Recall
    re = keras.metrics.Recall()
    re.update_state(labels, predictions)
    print("Recall: {0:.3f}".format(re.result().numpy()))
    print()
