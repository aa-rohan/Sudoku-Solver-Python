from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import pickle
from sigmoid import sigmoid
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

imageSize = 28
imagePixels = imageSize * imageSize

weights = pickle.load(open('weights.pkl', 'rb'))


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
    gray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def getContour(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleContour = None

    for c in cnts:
        if cv2.contourArea(c) > 4000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                puzzleContour = approx
    return puzzleContour


def findPuzzle(img):
    processedImg = preProcess(img)
    # cv2.imshow("Processed Image", processedImg)
    puzzleContour = getContour(processedImg)
    puzzle = four_point_transform(img, puzzleContour.reshape(4, 2))
    # puzzle = puzzle[10:puzzle.shape[0]-10, 7:puzzle.shape[1]-7]
    return puzzle


def notEmpty(cell):
    image = cv2.resize(cell, (28, 28))
    image = image[5:-5, 5:-5]
    image = np.array(image)
    variance = np.var(image)
    if variance > 2000:
        return True
    else:
        return False


def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def notInRow(arr, row):

    # Set to store characters seen so far.
    st = set()

    for i in range(0, 9):

        # If already encountered before,
        # return false
        if arr[row][i] in st:
            return False

        # If it is not an empty cell, insert value
        # at the current cell in the set
        if arr[row][i] != 0:
            st.add(arr[row][i])

    return True

# Checks whether there is any
# duplicate in current column or not.


def notInCol(arr, col):

    st = set()

    for i in range(0, 9):

        # If already encountered before,
        # return false
        if arr[i][col] in st:
            return False

        # If it is not an empty cell, insert
        # value at the current cell in the set
        if arr[i][col] != 0:
            st.add(arr[i][col])

    return True

# Checks whether there is any duplicate
# in current 3x3 box or not.


def notInBox(arr, startRow, startCol):

    st = set()

    for row in range(0, 3):
        for col in range(0, 3):
            curr = arr[row + startRow][col + startCol]

            # If already encountered before,
            # return false
            if curr in st:
                return False

            # If it is not an empty cell,
            # insert value at current cell in set
            if curr != 0:
                st.add(curr)

    return True

# Checks whether current row and current
# column and current 3x3 box is valid or not


def isValid(arr, row, col):

    return (notInRow(arr, row) and notInCol(arr, col) and
            notInBox(arr, row - row % 3, col - col % 3))


def isValidConfig(arr, n):

    for i in range(0, n):
        for j in range(0, n):

            # If current row or current column or
            # current 3x3 box is not valid, return false
            if not isValid(arr, i, j):
                return False

    return True


def possible(grid, y, x, n):
    for i in range(0, 9):
        if grid[y][i] == n:
            return False
    for i in range(0, 9):
        if grid[i][x] == n:
            return False
    x0 = (x//3)*3
    y0 = (y//3)*3
    for i in range(0, 3):
        for j in range(0, 3):
            if grid[y0+i][x0+j] == n:
                return False
    return True


def solve(grid, img):
    for y in range(9):
        for x in range(9):
            if grid[y][x] == 0:
                for n in range(1, 10):
                    if possible(grid, y, x, n):
                        grid[y][x] = n
                        solve(grid, img)
                        grid[y][x] = 0
                return
    for row in grid:
        print(row)
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            cv2.putText(img, str(grid[y][x]), (x*secW+int(secW/2)-10, int((y+0.8) * secH)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Puzzle Solution', img)


network = NeuralNetworkRun(imagePixels, 100, 10)


def select_image():
    path = filedialog.askopenfilename()
    if len(path) > 0:
        sudoku = cv2.imread(path)
        sudoku = cv2.resize(sudoku, (500, 500))
        # sudoku = sudoku[10:sudoku.shape[0]-10, 10:sudoku.shape[1]-10]
        cv2.imshow("Original Puzzle", sudoku)
        puzzleImage = sudoku.copy()
        puzzleImage = findPuzzle(puzzleImage)
        # cv2.imshow("Located Grid", puzzleImage)
        img_name = "sudokugrid.jpg"
        cv2.imwrite(img_name, puzzleImage)

        image = cv2.imread('sudokugrid.jpg')
        copy = image.copy()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

        copy = cv2.resize(copy, (486, 486))
        boxes = splitBoxes(copy)

        inc = 4
        for j in range(4):
            board = []
            sudoku_row = []
            ROI_number = 0
            for box in boxes:
                box = box[7:box.shape[0]-7, 7:box.shape[1]-7]
                image = box
                original = image.copy()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                if notEmpty(thresh):

                    # Find contours, obtain bounding box, extract and save ROI
                    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        ROI = thresh[y:y+h, x:x+w]
                        outputImage = cv2.copyMakeBorder(
                            ROI,
                            inc,
                            inc,
                            int((ROI.shape[0]-ROI.shape[1])/2)+inc,
                            int((ROI.shape[0]-ROI.shape[1])/2)+inc,
                            cv2.BORDER_CONSTANT,
                            value=[0, 0, 0]
                        )
                        cv2.imwrite('ROI/ROI_{}.png'.format(ROI_number), outputImage)
                else:
                    cv2.imwrite('ROI/ROI_{}.png'.format(ROI_number), thresh)
                ROI_number += 1

            count = 0
            for i in range(81):
                not_empty = False
                image = cv2.imread('ROI/ROI_{}.png'.format(i))
                image = cv2.resize(image, (28, 28))
                if notEmpty(image):
                    not_empty = True
                image = image*(0.99/255)
                image = image[:, :, 0]
                image = image.reshape(784)
                prediction = network.run(image)
                if not_empty and prediction == 0:
                    sudoku_row.append(8)
                elif not_empty:
                    sudoku_row.append(prediction)
                else:
                    sudoku_row.append(0)
                count += 1
                if count == 9:
                    board.append(sudoku_row)
                    sudoku_row = []
                    count = 0
            if(isValidConfig(board, 9)):
                break
            inc = inc + 1

        solve(board, copy)
        # for row in board:
        #     print(row)

        cv2.waitKey(0)


root = Tk()
btn = Button(root, text="Select a Puzzle", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
