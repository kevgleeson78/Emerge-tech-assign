# import keras for building the nural network
import keras as kr
# Import gzip for unpacking the images and labels
import gzip
# Import numpy
import numpy as np
# Import sklearn for categorising each digit
import sklearn.preprocessing as pre

from keras.preprocessing import image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
# The code in this script was mainly Adapted from: https://raw.githubusercontent.com/ianmcloughlin/jupyter-teaching-notebooks/master/mnist.ipynb
# Start a neural network, building it by layers.
model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784.
# There are 784 input neurons as this value is equal to the total amount of bytes each image has.
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))
# Add ten neurons to the output layer
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Open the gzipped files and read as bytes.
#Adapted from : https://docs.python.org/2/library/gzip.html
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()
# read in all images and labels into memory
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[8:])).astype(np.uint8)

# Flatten the array so the inputs can be mapped to the input neurons
inputs = train_img.reshape(60000, 784)
# encode the labels into binary format
encoder = pre.LabelBinarizer()
# get the size of the array needed for each category
encoder.fit(train_lbl)
# encode each label to be used as binary outputs
outputs = encoder.transform(train_lbl)
# print out the integer value and the new representation of the number
print(train_lbl[0], outputs[0])

# print out each array
for i in range(10):
    print(i, encoder.transform([i]))
def train_model():
    # Start the training
    # Set the model up by adding the input and output layers to the network
    #The epochs value is the amount of test runs are needed
    # The batch_size value is the amount of images sent at one time to the network
    model.fit(inputs, outputs, epochs=40, batch_size=100)

    model.save("data/my_model.h5")


def load_model():
    model.load_weights("data/my_model.h5")


# adapted from https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
option = input("Load model y/n?")
if option == 'y':
    load_model()
elif option == 'n':
    train_model()


# open the gzipped test images and labels
# Adapted from : https://docs.python.org/2/library/gzip.html
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()

# Store each image and label into memory
# Adapted from: https://raw.githubusercontent.com/ianmcloughlin/jupyter-teaching-notebooks/master/mnist.ipynb
test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

# Print out the performance of the network
performance = (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()
print("The correct number of predictions", performance)
## Get 20 random images form the test set
# Random int adapted from https://stackoverflow.com/questions/3996904/generate-random-integers-between-0-and-9
from random import randint
for i in range(20):
    print("Test Number : ", i+1,"\n")
    x = randint(0, 9999)
    print("The random index: ", x, "\n")
    print("The result array: ")
    test = model.predict(test_img[x:x+1])
    # Print the result array
    print(test, "\n")
    # Get the maximum value from the machine predictions
    pred_result = test.argmax(axis=1)

    print("The machine prediction is : =>> ",  pred_result)
    print("The actual number is : =>> ", test_lbl[x:x+1])
    print("##############################################")


# Testing other images
# input file adapted from https://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python
def file_upload():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    # Adapted from https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b
    img = image.load_img(path=file_path,color_mode = "grayscale",target_size=(28,28,1))

    imgage1 = np.array(list(image.img_to_array(img))).reshape(1, 784).astype(np.uint8) / 255.0
    plt.imshow(img)
    plt.show()
    test1 = model.predict(imgage1)

    print("The number predicted is : ", test1.argmax(axis=1))


upload_img = input("upload image? y/n")
if upload_img == 'y':
    file_upload()
elif upload_img == 'n':
    exit()