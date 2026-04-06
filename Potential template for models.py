import sys
import tensorflow as tf
from keras import layers
import keras as keras
import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
#here they would have scikit and import train_test_split i dont think we need it


#----fetching data----- (may be another file)
#TODO

# -------Defining the model ------------ 

#From book these keras calls might need to be layers for our current import without 
#line 4
CNN = keras.Sequential(
    [
        keras.layers.Input(shape=(784,)),
        # Hidden layer 1 = 256 nodes, linear activation
        keras.layers.Dense(256, activation="linear"),
        # Hidden layer 2: 128 nodes, linear activation
        keras.layers.Dense(128, activation="linear"),
        # Output layer: 10 nodes, one per class
        keras.layers.Dense(10, activation="softmax"),
    ]
)
CNN.summary()


# -------Training the model ------------ Table 6.4.3
#TODO
# model.compile(
#     #all are example choices use is not required
#     optimizer = "rmsprop",
#     loss = "SparseCategoricalCrossentropy",
#     metrics=["accuracy"],
# )


#model.fit(Xtrain, ytrain, batch_size=64, epochs=10)
# ^ should be all we need

# Can do the below code to help with evaluation but we could honnestly try the code below that 
# Book says: training =model.fit(Xtrain, ytrain, batch_size=64, epochs=10) \n  training.history
#We could do: model.history


# -------Evaluating the model ------------ 
#TODO
#results = model.evaluate(Xtest, ytest, batch_size=64)
#print("Test loss, accuracy", results)
# predictions = model.predict(Xtest[:3])
# print("Predictions:", predictions.round(3))
# print("Actual values:", ytest[:3])




print("Congrats it works")