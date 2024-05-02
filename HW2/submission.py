#!/usr/bin/python

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """
    # BEGIN_YOUR_ANSWER
    return {'so': 0, 'interesting':0,'great':1,'plot':1,'bored':-1,'not':-1}
    raise NotImplementedError # replace this line with your code
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    words = x.split()
    features = {}

    for word in words:
        if word in features:
            features[word] += 1
        else:
            features[word] = 1

    return features
    raise NotImplementedError # replace this line with your code
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER
    for t in range(numIters):
        for x, y in trainExamples:
            score = 0
            for feature, value in featureExtractor(x).items():
                if feature in weights:
                    score += weights[feature]*value

            for feature, value in featureExtractor(x).items():
                if y==1: gradient = (1-sigmoid(score))*value
                else: gradient = -sigmoid(score)*value

                if feature not in weights:
                    weights[feature] = 0
                weights[feature] += eta*gradient

    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    words = x.split()
    n_feature = {}

    for i in range(len(words)-n+1):
        word = words[i]
        for j in range(n-1):
            word += " " + words[i+j+1]
        n_feature[word] = n_feature.get(word, 0) + 1

    return n_feature
    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        def sigmoid(n):
            return 1 / (1 + np.exp(-n))
        
        self.x =x
        self.f1 = x@self.W1 + self.b1
        self.f2 = sigmoid(self.f1)
        self.f3 = self.f2@self.W2 + self.b2
        self.f4 = sigmoid(self.f3)

        # print("x is")
        # print(x)
        # print("self.W2 is")
        # print(self.W2)
        # print("f1 is")
        # print(self.f1)
        # print()

        return self.f4.squeeze()
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER
        loss = np.zeros_like(pred)
        for i in range(len(target)):
            if target[i] == 0:
                loss[i]=-math.log(1-pred[i])
            else:
                loss[i]=-math.log(pred[i])
        
        return loss
        raise NotImplementedError # replace this line with your code
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER
        # Store the gradients in a dictionary
        gradients={}
        for i in range(len(target)):
            if target[i] == 0:
                self.f4[i]=self.f4[i]
            else:
                self.f4[i]=self.f4[i]-1

        db2 = np.array([[np.sum(self.f4)]])
        dw2 = self.f2.T@self.f4

        db1 = self.f4@self.W2.T
        df3=self.f2*(1-self.f2)
        db1 = db1*df3

        dw1 = self.x.T@db1
        db1 = np.sum(db1,axis=0,keepdims=True)

        gradients["W1"] = dw1
        gradients["b1"] = db1
        gradients["W2"] = dw2
        gradients["b2"] = db2

        return gradients
        raise NotImplementedError # replace this line with your code
        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        self.W1 = self.W1 - gradients["W1"]*learning_rate
        self.b1 = self.b1 - gradients["b1"]*learning_rate
        self.W2 = self.W2 - gradients["W2"]*learning_rate
        self.b2 = self.b2 - gradients["b2"]*learning_rate

        return
        raise NotImplementedError # replace this line with your code
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER
        for _ in range(epochs):
            for i in range(X.shape[0]):

                pred_ = self.forward(X[i:i+1])
                pred = np.array([pred_])

                loss = self.loss(pred,Y[i:i+1])

                gradient = self.backward(pred, Y[i:i+1])
                self.update(gradient, learning_rate)                

        return loss
        raise NotImplementedError # replace this line with your code
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))