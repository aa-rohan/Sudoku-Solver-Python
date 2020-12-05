import numpy as np
import pickle
from sigmoid import sigmoid

data = pickle.load(open("pickled_mnist.pkl", "br"))

trainImgs = data[0]
testImgs = data[1]
trainLabelsOneHot = data[4]
testLabelsOneHot = data[5]

imageSize = 28
imagePixels = imageSize * imageSize


class NeuralNetwork:
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, learningRate):

        self.weights1 = np.random.randn(numHiddenNodes, numInputNodes)
        self.weights2 = np.random.randn(numOutputNodes, numHiddenNodes)
        self.learningRate = learningRate

    def train(self, input, target):
        input = np.array(input, ndmin=2).T
        target = np.array(target, ndmin=2).T

        output1 = sigmoid(np.dot(net.weights1, input))
        finalOutput = sigmoid(np.dot(net.weights2, output1))

        costOutput = target - finalOutput

        deltaOutput = finalOutput*(1.0-finalOutput)*costOutput
        self.weights2 += self.learningRate*np.dot(deltaOutput, output1.T)

        costHidden = np.dot(self.weights2.T, costOutput)
        deltaHidden = output1*(1.0-output1)*costHidden
        self.weights1 += self.learningRate*np.dot(deltaHidden, input.T)

    def run(self, input):
        input = np.array(input, ndmin=2).T
        output = sigmoid(np.dot(self.weights2, sigmoid(np.dot(self.weights1, input))))
        return output

    def test(self, input, target):
        correct = 0
        tested = 0
        for i in range(len(input)):
            inp = np.array(input[i], ndmin=2).T
            out = sigmoid(np.dot(self.weights2, sigmoid(np.dot(self.weights1, inp))))
            if np.argmax(out) == np.argmax(target[i]):
                correct += 1
            tested += 1
            cor = str(correct)
            tes = str(tested)
            print(cor+"/"+tes)
        return correct/tested

    def multitrain(self, input, target, epochs):
        intermediateWeights = []
        maxAccuracy = 0
        for epoch in range(epochs):
            for i in range(len(input)):
                self.train(input[i], target[i])
            newAccuracy = self.test(testImgs, testLabelsOneHot)
            if newAccuracy > maxAccuracy:
                intermediateWeights.append((self.weights1.copy(), self.weights2.copy()))
                maxAccuracy = newAccuracy
            print("Epoch: " + str(epoch) + " Accuracy: " + str(maxAccuracy))
        return intermediateWeights


net = NeuralNetwork(imagePixels, 100, 10, 0.15)

weights = net.multitrain(trainImgs, trainLabelsOneHot, 25)

pickle.dump(weights, open('weights.pkl', 'wb'))
