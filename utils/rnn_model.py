#import csv
#import pickle
#import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#import os
from tensorflow.keras import layers


# try out the labmda defining the input shape to have variable dimensions on an axis
# try out bidirectional lstm layer
# convert each recording into an a 3d image


def model(MODEL_INPUT_SHAPE, dp1=0, dp2=0, dp3=0, dp4=0, learning_rate=1e-05,LOSS=tf.keras.losses.Huber(), NUMBER_CLASSES=4):
    """
        Create a Convolutional Neural Network (CNN) with Long Short-Term Memory (LSTM) layers for classification.
    
        Parameters:
        - MODEL_INPUT_SHAPE: tuple, shape of the input data.
        - dp1, dp2, dp3, dp4: float, dropout rates for different layers.
        - learning_rate: float, learning rate for the Adam optimizer.
        - LOSS: loss function, default is Huber loss.
        - NUMBER_CLASSES: int, number of output classes.
    
        Returns:
        - tf.keras.Model: compiled model for training and evaluation.
    """
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # (1e-05 == 0.00001)

    # dp1 = dropout rate 1
    input_shape = MODEL_INPUT_SHAPE
    input_layer = tf.keras.layers.Input(input_shape)

    layer_c1 = layers.Conv1D(
        filters=32,
        kernel_size=(3),
        padding="same",
        activation="relu",
        kernel_initializer="HeUniform",
        name="l1",
    )(input_layer)
    layer_c1_b = layers.BatchNormalization(epsilon=0.001)(layer_c1)
    layer_c1_m = layers.MaxPooling1D(padding="same")(layer_c1_b)  # ADDED
    layer_d4 = layers.Dropout(dp4)(layer_c1_m)

    layer_c2 = layers.Conv1D(
        filters=64,
        kernel_size=(3),
        padding="same",
        activation="relu",
        kernel_initializer="HeUniform",
        name="l2",
    )(layer_d4)
    layer_c2_b = layers.BatchNormalization(epsilon=0.001)(layer_c2)
    layer_c2_m = layers.MaxPooling1D(padding="same")(layer_c2_b)  # ADDED
    layer_d3 = layers.Dropout(dp1)(layer_c2_m)
   

    layer_c3 = layers.Conv1D(
        filters=128,
        kernel_size=(3),
        padding="same",
        activation="relu",
        kernel_initializer="HeUniform",
        name="l3",
    )(layer_d3)
    layer_c3_b = layers.BatchNormalization(epsilon=0.001)(layer_c3)
    layer_c3_m = layers.MaxPooling1D(padding="same")(layer_c3_b)  # ADDED
    layer_d1 = layers.Dropout(dp2)(layer_c3_m)


    # second
    layer_l1 = layers.LSTM(units=32, return_sequences=True)(layer_d1)
    layer_d2 = layers.Dropout(dp3)(layer_l1)
   
    """
    #second
    layer_l1 = layers.LSTM(units = 32, return_sequences = True)(input_layer)
    layer_l2 = layers.LSTM(units = 512, return_sequences = True)(layer_l1) #this reached 100 acc on same trained set---perfectly overfitted
    layer_l1_b1 = layers.BatchNormalization(epsilon = 0.001)(layer_l2) ###ADDED
    layer_d2 = layers.Dropout(dp3)(layer_l1_b1)
    """

    #layer_conc = layers.concatenate([layer_l1, layer_c3_m])
    layer_conc = layers.concatenate([layer_d2, layer_d1])

    l1 = layers.Flatten()(layer_conc)
    l2 = layers.Dense(64, activation="softmax")(l1)
    l3 = layers.Dense(NUMBER_CLASSES)(l2)

    model1 = tf.keras.Model(inputs=input_layer, outputs=l3)
    model1.compile(
        loss=LOSS,
        optimizer=optimiser,
        metrics=["accuracy"],
    )

    # model1.summary()
    return model1
