import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
with open ('data/RNNhistoryB64E35L7.json', 'r') as file:
    data = json.load(file)
dataOneName = 'RNNhistoryB64E35L7'
dataTwoName = 'CNNhistoryB64E18L7'
toCompare = 'accuracy'

RNNdf = pd.read_json('data/'+ dataOneName + '.json')
CNNdf = pd.read_json('data/' + dataTwoName + '.json')

xRNN = range(1, len(RNNdf) + 1)
xCNN = range(1, len(CNNdf) + 1)

plt.plot(xRNN, RNNdf[toCompare], label='RNN')
plt.plot(xCNN, CNNdf[toCompare], label='CNN')

plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('CNN vs RNN accuracy over time')
plt.legend()

plt.savefig(dataOneName + 'Vs' + dataTwoName + str.capitalize(toCompare))