import sys
import tensorflow as tf
from keras import layers, datasets
import keras as keras
import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" #Silences the voices
#here they would have scikit and import train_test_split i dont think we need it


#----fetching data----- (may be another file)

(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.cifar10.load_data() 
#^ our data is premade so we might not even have to load it oursleves just laod from the library


# -------Defining the model ------------ 

#From book these keras calls might need to be layers for our current import without 
#line 4
#TODO Rename all cnn to something else probably RCN

CNN = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),

    keras.layers.Conv2D(256, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(512, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
CNN.summary()


# -------Training the model ------------ Table 6.4.3
CNN.compile(
     optimizer = "adam",
     loss = "SparseCategoricalCrossentropy",
     metrics=["accuracy"],
 )

CNN.fit(Xtrain, ytrain, batch_size=128, epochs=18, validation_split=0.2)


# Can do the below code to help with evaluation but we could honnestly try the code below that 
# Book says: training =model.fit(Xtrain, ytrain, batch_size=64, epochs=10) \n  training.history
#We could do: model.history


# -------Evaluating the model ------------ 
results = CNN.evaluate(Xtest, ytest, batch_size=128)
print("Test loss, accuracy", results)
predictions = CNN.predict(Xtest[:10])
predicted_classes = np.argmax(predictions, axis=1)
print("Predictions:", predicted_classes)
print("Actual values:", ytest[:10]) #We probably wont have an actual values since we don't have labels for test




print("Everything has run :)")