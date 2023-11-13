import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow import keras

METRICS = [
    # keras.metrics.TruePositives(name='tp'),
    # keras.metrics.FalsePositives(name='fp'),
    # keras.metrics.TrueNegatives(name='tn'),
    # keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

def make_model(layers, lr, metrics=METRICS):
    model = keras.Sequential()
    model.add(keras.layers.Dense(layers[1], activation="sigmoid", input_shape=(layers[0],)))

    for layer in layers[2:]:
        model.add(keras.layers.Dense(layer, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)   
    return model

def make_model_with_dropout(layers, lr, metrics=METRICS):
    model = keras.Sequential()
    model.add(keras.layers.Dense(layers[1], activation="sigmoid", input_shape=(layers[0],)))
    if len(layers)>2:
        model.add(keras.layers.Dropout(0.5))
    
    for layer in layers[2:-1]:
        model.add(keras.layers.Dense(layer, activation="sigmoid"))
        model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(layers[-1], activation="sigmoid"))
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)   
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, stop_early):
    print("Stop early:", stop_early)
    print(model.summary())

    # Calculate class weights
    pos = np.sum(y_train)
    neg = len(y_train) - pos
    total = pos + neg
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    if stop_early:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=2,
            patience=epochs,
            mode='min',
            restore_best_weights=True)
        callbacks = [early_stopping]
    else:
        callbacks = None

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        callbacks = callbacks,
        class_weight=class_weight,
        validation_data=(X_val, y_val),
        verbose=2)

    return model, history

def eval_model(model, X, y, text=None):
    if text is not None:
        print(text)
    results = model.evaluate(X, y, verbose=2)
    for name, value in zip(model.metrics_names, results):
        print(name, ': ', value)

    preds = model.predict(X)
    ba = metrics.balanced_accuracy_score(y, preds>0.5)
    print("balanced accuracy :", ba)
    print()
