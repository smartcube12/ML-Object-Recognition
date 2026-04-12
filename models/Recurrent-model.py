import sys
import tensorflow as tf
import math
from keras import layers, datasets
from sklearn import metrics
import keras as keras
import numpy as np
import pandas as pd
import os
import json
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" #Silences the voices
#here they would have scikit and import train_test_split i dont think we need it


#----fetching data----- (may be another file)

(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.cifar10.load_data() 
ytrain = tf.keras.utils.to_categorical(ytrain, 10)
ytest = tf.keras.utils.to_categorical(ytest, 10)
#^ our data is premade so we might not even have to load it oursleves just laod from the library


# -------Defining the model ------------ 

#From book these keras calls might need to be layers for our current import without 
#line 4
#TODO Rename all cnn to something else probably RNN

Xtrain = Xtrain.reshape((-1, 32, 32*3))
Xtest = Xtest.reshape((-1, 32, 32*3))

RNN = keras.Sequential([
    keras.layers.Input(shape=(32, 96)),

    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),

    layers.LSTM(128),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
RNN.summary()


# -------Training the model ------------ Table 6.4.3
RNN.compile(
     optimizer = "adam",
     loss = "CategoricalCrossentropy",
     metrics=["accuracy",
        'precision',
        'recall'],
 )

RNNModel = RNN.fit(Xtrain, ytrain, batch_size=64, epochs=35, validation_split=0.4)


# Can do the below code to help with evaluation but we could honnestly try the code below that 
# Book says: training =model.fit(Xtrain, ytrain, batch_size=64, epochs=10) \n  training.history
#We could do: model.history


# -------Evaluating the model ------------ 

with open('output/RNNhistoryB64E35L7.json', 'w') as f:
    json.dump(RNNModel.history, f)

results = RNN.evaluate(Xtest, ytest, batch_size=64)
print("Test loss, accuracy", results)
predictions = RNN.predict(Xtest[:10])
predictionsStats = RNN.predict(Xtest)
predicted_classes = np.argmax(predictions, axis=1)
pStats = np.argmax(predictionsStats, axis=1)
ytestForMatrix = np.argmax(ytest, axis=1)
print("Predictions:", predicted_classes)
print("Actual values:", ytest[:10]) #We probably wont have an actual values since we don't have labels for test
confusionMatrix = metrics.confusion_matrix(ytestForMatrix, pStats, normalize="pred")
print("Confusion Matrix\n", confusionMatrix)
print("accuracy:", metrics.accuracy_score(np.ravel(ytestForMatrix), pStats))
print("precision:", metrics.precision_score(ytestForMatrix, pStats, average = "macro"))
print("recall:", metrics.recall_score(ytestForMatrix, pStats, average="macro"))
print("kappa:", metrics.cohen_kappa_score(ytestForMatrix, pStats))


acc = metrics.accuracy_score(ytestForMatrix, pStats)
n = len(ytestForMatrix)
se = math.sqrt((acc * (1 - acc)) / n)
lower = acc - 1.96 * se
upper = acc + 1.96 * se
print(f"95% Confidence Interval: ({lower:.4f}, {upper:.4f})")

print("Everything has run :)")