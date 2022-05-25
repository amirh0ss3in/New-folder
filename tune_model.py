from p import xtest, ytest, xtrain, ytrain
import kerastuner as kt
import tensorflow as tf 
import numpy as np
from kerastuner import HyperModel, BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix
import os

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.compat.v1.Session(config=config)

print(kt.__version__)
print(xtest.shape,ytest.shape,xtrain.shape,ytrain.shape)
cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

# https://keras.io/guides/keras_tuner/distributed_tuning/
# https://towardsdatascience.com/hyperparameter-tuning-with-keras-tuner-283474fbfbe

input_shape = xtrain.shape[1:]

def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for i in range(hp.Int("conv_layers", 4, 7, default=6)):
     
        x = tf.keras.layers.Conv1D(
            filters=hp.Int("filters_" + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int("kernel_size_" + str(i), 3, 6),
            activation='relu',
            padding="valid",
        )(x)

        if hp.Choice("pooling" + str(i), ["max", "avg"]) == "max":
            x = tf.keras.layers.MaxPooling1D()(x)
        else:
            x = tf.keras.layers.AveragePooling1D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice("global_pooling", ["max", "avg"]) == "max":
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice("optimizer", ["adam", "sgd"])
    model.compile(
        optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model



tuner_bo = BayesianOptimization(
            hypermodel = build_model,
            objective='val_accuracy',
            max_trials=40,
            seed=42,
            project_name=cwd+'new3'
        )

tuner_bo.search(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=2, epochs=140, verbose=1)
best_model = tuner_bo.get_best_models(num_models=1)[0]
best_model.evaluate(xtest, ytest)

ytest_pred = best_model.predict(xtest)
ytest_pred = np.argmax(ytest_pred, axis=1)
print(ytest_pred.shape)
print(confusion_matrix(ytest,ytest_pred))
print(classification_report(ytest,ytest_pred,digits=4))