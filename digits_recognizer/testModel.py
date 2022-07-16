
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import struct


test_images = None
test_labels = None

basaPath = "/Users/claudiomiraka/Desktop/tests/ai_tests/"

with open(basaPath+'mnist/t10k-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_images = data.reshape((size, nrows, ncols))

with open(basaPath+'mnist/t10k-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    test_labels = data.reshape((size, 1))

with open(basaPath+'mnist/train-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_labels = data.reshape((size, 1))
    

def showImage(guessed, index):
    p = np.max(test_labels[index])
    actual = np.where(test_labels[index] ==  p)[0][0]
    print("guess: "+str(guessed) + " actual: " + str(actual))
    plt.title("guess: "+str(guessed) + " actual: " + str(actual))
    plt.imshow(test_images[index,:,:], cmap='gray')
    plt.show()

num_classes = 10
test_labels = np_utils.to_categorical(test_labels, num_classes)

model = tf.keras.models.load_model('my_model')
model.summary()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


pred = model.predict(test_images)

while True:
    index = np.random.randint(0, len(test_labels))
    p = np.max(pred[index])
    guess = np.where(pred[index] ==  p)
    showImage(guess[0][0], index)
