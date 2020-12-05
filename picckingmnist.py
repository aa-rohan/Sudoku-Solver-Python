import numpy as np
import pickle

imageSize = 28
numLabels = 10

train = np.empty([60000, 785])
test = np.empty([10000, 785])

row = 0
for line in open("mnist_train.csv"):
    train[row] = np.fromstring(line, sep=",")
    row += 1

row = 0
for line in open("mnist_test.csv"):
    test[row] = np.fromstring(line, sep=",")
    row += 1

fac = 0.99 / 255
trainImgs = np.asfarray(train[:, 1:]) * fac + 0.01
testImgs = np.asfarray(test[:, 1:]) * fac + 0.01

trainLabels = np.asfarray(train[:, :1])
testLabels = np.asfarray(test[:, :1])


lr = np.arange(10)

for label in range(10):
    oneHot = (lr == label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", oneHot)


lr = np.arange(numLabels)

# transform labels into one hot representation
trainLabelsOneHot = (lr == trainLabels).astype(np.float)
testLabelsOneHot = (lr == testLabels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
trainLabelsOneHot[trainLabelsOneHot == 0] = 0.01
trainLabelsOneHot[trainLabelsOneHot == 1] = 0.99
testLabelsOneHot[testLabelsOneHot == 0] = 0.01
testLabelsOneHot[testLabelsOneHot == 1] = 0.99

with open("pickled_mnist.pkl", "bw") as fh:
    data = (trainImgs,
            testImgs,
            trainLabels,
            testLabels,
            trainLabelsOneHot,
            testLabelsOneHot)
    pickle.dump(data, fh)
