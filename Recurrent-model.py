import sys
import tensorflow as tf
import math
from keras import layers, datasets
from sklearn import metrics
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

Xtrain = Xtrain.reshape((-1, 32, 32*3))
Xtest = Xtest.reshape((-1, 32, 32*3))

RCN = keras.Sequential([
    keras.layers.Input(shape=(32, 96)),
    keras.layers.LSTM(512),
    keras.layers.Dense(10, activation="softmax"),
])
RCN.summary()


# -------Training the model ------------ Table 6.4.3
RCN.compile(
     optimizer = "adam",
     loss = "SparseCategoricalCrossentropy",
     metrics=["accuracy"],
 )

RCN.fit(Xtrain, ytrain, batch_size=256, epochs=1, validation_split=0.4)


# Can do the below code to help with evaluation but we could honnestly try the code below that 
# Book says: training =model.fit(Xtrain, ytrain, batch_size=64, epochs=10) \n  training.history
#We could do: model.history


# -------Evaluating the model ------------ 
results = RCN.evaluate(Xtest, ytest, batch_size=128)
print("Test loss, accuracy", results)
predictions = RCN.predict(Xtest[:10])
predictionsStats = RCN.predict(Xtest)
predicted_classes = np.argmax(predictions, axis=1)
pStats = np.argmax(predictionsStats, axis=1)
ytestForMatrix = ytest.flatten()
print("Predictions:", predicted_classes)
print("Actual values:", ytest[:10]) #We probably wont have an actual values since we don't have labels for test
confusionMatrix = metrics.confusion_matrix(ytestForMatrix, pStats, normalize="pred")
print("Confusion Matrix\n", confusionMatrix)
print("accuracy:", metrics.accuracy_score(np.ravel(ytest), pStats))
print("precision:", metrics.precision_score(ytest, pStats, average = "macro"))
print("recall:", metrics.recall_score(ytest, pStats, average="macro"))
print("kappa:", metrics.cohen_kappa_score(ytest, pStats))


acc = metrics.accuracy_score(ytestForMatrix, pStats)
n = len(ytestForMatrix)
se = math.sqrt((acc * (1 - acc)) / n)
lower = acc - 1.96 * se
upper = acc + 1.96 * se
print(f"95% Confidence Interval: ({lower:.4f}, {upper:.4f})")

print("Everything has run :)")