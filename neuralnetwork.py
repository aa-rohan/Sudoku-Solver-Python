import numpy as np
import pickle
import cv2
from sigmoid import sigmoid

weights = pickle.load(open('weights.pkl', 'rb'))


imageSize = 28
imagePixels = imageSize * imageSize

width = 640
height = 480


class NeuralNetworkRun:
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes):

        self.weights1 = weights[7][0]
        self.weights2 = weights[7][1]

    def run(self, input):
        input = np.array(input, ndmin=2).T
        output = sigmoid(np.dot(self.weights2, sigmoid(np.dot(self.weights1, input))))
        return np.argmax(output)

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


def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 57, 5)
    # img = cv2.Canny(img, 30, 50)
    # img = cv2.equalizeHist(img)
    img = img*(0.99/255)
    return img


network = NeuralNetworkRun(imagePixels, 100, 10)

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    # Resizing into 128x128 because we trained the model with this image size.
    im = cv2.resize(frame, (28, 28))
    im = preProcess(im)
    imgArray = np.array(im)
    imgArray = imgArray.reshape(784)

    prediction = network.run(imgArray)

    cv2.putText(frame, "Prediction: " + str(prediction), (50, 50),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
