from matplotlib import pyplot as plt
import numpy as np
import struct
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Conv1D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling1D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras import backend as K
from keras.initializers import VarianceScaling


train_images = None
train_labels = None

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

with open(basaPath+'mnist/train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_images = data.reshape((size, nrows, ncols))

with open(basaPath+'mnist/train-labels-idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    train_labels = data.reshape((size, 1))

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        #self.keys = ['loss', 'acc', 'val_loss', 'val_acc']
        self.keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        self.values = {}
        for k in self.keys:
            self.values['batch_'+k] = []
            self.values['epoch_'+k] = []

    def on_batch_end(self, batch, logs={}):
        for k in self.keys:
            bk = 'batch_'+k
            if k in logs:
                self.values[bk].append(logs[k])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.keys:
            ek = 'epoch_'+k
            if k in logs:
                self.values[ek].append(logs[k])

    def plot(self, keys):
        for key in keys:
            plt.plot(np.arange(len(self.values[key])), np.array(self.values[key]), label=key)
        plt.legend()

def run_keras(X_train, y_train, X_test, y_test, layers, epochs, split=0, verbose=True):
    # Model specification
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Define the optimization
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])
    N = X_train.shape[0]
    # Pick batch size
    batch = 32 if N > 1000 else 1     # batch size
    history = LossHistory()
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=split, callbacks=[history], verbose=verbose)

    # Evaluate the model on validation data, if any
    # if X_val is not None or split > 0:
    #     val_acc, val_loss = history.values['epoch_val_accuracy'][-1], history.values['epoch_val_loss'][-1]
    #     print ("\nLoss on validation set:"  + str(val_loss) + " Accuracy on validation set: " + str(val_acc))
    # else:
    #     val_acc = None
    # Evaluate the model on test data, if any
    if X_test is not None:
        test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch)
        print ("\nLoss on test set:"  + str(test_loss) + " Accuracy on test set: " + str(test_acc))
    else:
        test_acc = None
    return model, history, test_acc

def showImage(guessed, index):
    p = np.max(test_labels[index])
    actual = np.where(test_labels[index] ==  p)[0][0]

    print( str(guessed)+ " " + str(actual))
    plt.title("guess: "+str(guessed) + " actual: " + str(actual))
    plt.imshow(test_images[index,:,:], cmap='gray')
    plt.show()

layers = [
    Conv2D(
        filters=32, 
        kernel_size=(3,3),
        activation='relu',
        input_shape=(28 ,28,1),
    ),
    MaxPooling2D(pool_size=(2,2)),
        Conv2D(
        filters=64, 
        kernel_size=(3,3),
        activation='relu',
    ),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(units=128, activation="relu"),
    # Dropout(0.5),
    Dense(units=10, activation="softmax")
]


epochs = 1 
m = train_images.shape[1]
train_images = train_images.reshape((train_images.shape[0], m, m, 1))
test_images = test_images.reshape((test_images.shape[0], m, m, 1))

num_classes = 10
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

(model, history, test_acc) = run_keras( train_images, train_labels,test_images, test_labels, layers, epochs)

model.summary()

model.save('my_model')

print("Model saved!")

