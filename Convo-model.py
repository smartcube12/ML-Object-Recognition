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



# trainCSV = pd.read_csv("possibly_better_labels.csv")
# trainCSV["filename"] = "input/train" + trainCSV["id"]
# labels = trainCSV["label"].astype("category")
# trainCSV["label_num"] = labels.cat.codes
# filepaths = trainCSV["filename"].values
# labels = trainCSV["label_num"].values

# dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.cifar10.load_data() 
#^ our data is premade so we might not even have to load it oursleves just laod from the library

#print(trainCSV.head(10))  #Testing if w are reading training labels

#----separating data----- (may be another file)

# Training = tf.keras.preprocessing.image_dataset_from_directory(
#     "input/train",
#     image_size=(32,32),
#     batch_size=50000
# )

# Testing = tf.keras.preprocessing.image_dataset_from_directory(
#     "input/test",
#     image_size=(32,32),
#     batch_size=300000
# )

# def load_image(path, label):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.resize(img, (32, 32))
#     img = img / 255.0
#     return img, label

# dataset = dataset.map(load_image).batch(32)

# -------Defining the model ------------ 

#From book these keras calls might need to be layers for our current import without 
#line 4
CNN = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),

    keras.layers.Conv2D(128, (3,3), activation="relu"), #test vlaue 16, real vlaue 128
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(256, (3,3), activation="relu"), #test vlaue 32, real vlaue 256
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation="relu"), #test vlaue 16, real vlaue 128
    keras.layers.Dense(10, activation="softmax"),
])
CNN.summary()


# -------Training the model ------------ Table 6.4.3

CNN.compile(
     optimizer = "adam",
     loss = "SparseCategoricalCrossentropy",
     metrics=["accuracy"],
 )

CNN.fit(Xtrain, ytrain, batch_size=64, epochs=1, validation_split= 0.4) #test Batch vlaue 8, real vlaue 64. Real epochs 18
# ^ should be all we need

# Can do the below code to help with evaluation but we could honnestly try the code below that 
# Book says: training =model.fit(Xtrain, ytrain, batch_size=64, epochs=10) \n  training.history
#We could do: model.history


# -------Evaluating the model ------------ 

results = CNN.evaluate(Xtest, ytest, batch_size=64)
print("Test loss, accuracy", results)
predictions = CNN.predict(Xtest[:10])
predictionsStats = CNN.predict(Xtest)
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