import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
print(data["intents"])

words = []
labels = []
docs_x = []
docs_y = []

#stop prepocessing multiple times?stop rurunning code
try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f) 
except:
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

    with open('data.pickle', 'wb') as f:
       pickle.dump ((words, labels, training, output), f)

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

# stop training of model if a model already exist
try:
    model.load('model.tflearn')
except:
#pass training data
#number epoch = number of times NN sees same data
    model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
    model.save('model.tflearn')

#classifying model turn user input into BoW
def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    # stem words
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    #loop to generate words
    for sentence in s_words:
        for i, w in enumerate(words):
            if w == sentence:
                bag[i] = (1)
    return numpy.array(bag)

# ask user for input
def chat():
    print("Let's chat! (type quit to end chat)")
    while True:
        inp = input("You: ")
        #prevent loop from running infinitely
        if inp.lower() == "quit":
            break
        #turn input into BoW, feed into model, get model response --- function
        results = model.predict([bag_of_words(inp, words)])
       
       print results()
chat()