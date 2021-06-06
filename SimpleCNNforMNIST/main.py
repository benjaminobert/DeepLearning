from keras import layers
from keras import models

from keras.datasets import mnist
from keras.utils import to_categorical


class CNNModel:
    def __init__(self):
        self.model = models.Sequential()
        self.test_acc = []
        self.test_acc = []

    def PrintModelSummary(self):
        self.model.summary()

    def SetupCNNModel(self):
        print('+++Setup CNN based on Conv2D and MaxPooling2d layers ...')
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.PrintModelSummary()

        print("+++transform 3D result to a 1D vector+++")
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.PrintModelSummary()

    def CompileModel(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def FitModelToTrainingData(self, train_images, train_labels):
        self.model.fit(train_images, train_labels, epochs=5, batch_size=64)

    def CalculateLossAndAccuracy(self, test_images, test_labels):
        self.test_loss, self.test_acc = self.model.evaluate(test_images, test_labels)

def PrepareMNISTDataSet():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    print('+++ Simple Convolutional Neuronal Network to learn the fabolous MNIST dataset! +++')

    CNNModel = CNNModel()
    CNNModel.SetupCNNModel()
    CNNModel.CompileModel()

    train_images, train_labels, test_images, test_labels = PrepareMNISTDataSet()

    CNNModel.FitModelToTrainingData(train_images, train_labels)
    CNNModel.CompileModel()
    CNNModel.CalculateLossAndAccuracy(test_images, test_labels)

    print(f"Accuracy of {CNNModel.test_acc} and Loss of {CNNModel.test_loss} achieved for the MNIST test data set.")