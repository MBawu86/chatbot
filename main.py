import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)
print(data["intents"])

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent ['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

#sorts and lowercases input
words = [stemmer.stem(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))

labels = sorted(labels)

# input must be changed from string to number (bag of words aka 'one hot encoded')
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

#changes output to OHE/BoW
for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w)for w in doc]

#go through words in doc 

for w in words:
    #if word exist in currentpattern being looped thru
    if w in wrds:
        #yes word exists
        bag.append(1)
        #no word in not here
    else:
        bag.append(0)

        #append into trainng and output

    output_row = out_empty[:]
    output_row [labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

#reset data graph
tensorflow.reset_default_graph()

# nueral network
# input nuerons are length imput data (how many words I have)
net = tflearn.input_data(shape=[None, len(training[0])])
# 2 hidden networks with 8 nuerons each fully connected
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
#output layer w/softmax activation give probabilty to eac nueron grabs highest % / probabilty of appropriate chatbot response
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#pass training data
#number epoch = number of times NN sees same data
model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
model.save('model.tflearn')
