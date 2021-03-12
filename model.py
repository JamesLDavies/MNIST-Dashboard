"""
Module docstring

Usage:
* pip3 install -r requirements.txt
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import time
import logging

LOGGER = logging.getLogger(__name__)


NUM_CLASSES = 10


def _prep_data(train, test):
    """

    Args:
        train:
        test:

    Returns:
        train_norm
        test_norm
    """
    LOGGER.info("Preprocessing...")

    train = train.reshape(train.shape[0], 28, 28, 1)
    test = test.reshape(test.shape[0], 28, 28, 1)

    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


def _define_model():
    """

    Returns:

    """
    LOGGER.info("Defining model...")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def larger_model(num_classes):
    """

    Returns:

    """
    # create model
    model = Sequential()

    # Convolutional layer with 30 feature maps of size 5×5.
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # Pooling layer taking the max over 2*2 patches.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional layer with 15 feature maps of size 3×3.
    model.add(Conv2D(15, (3, 3), activation='relu'))

    # Pooling layer taking the max over 2*2 patches.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer with a probability of 20%.
    model.add(Dropout(0.2))

    # Flatten layer.
    model.add(Flatten())

    # Fully connected layer with 128 neurons and rectifier activation.
    model.add(Dense(128, activation='relu'))

    # Fully connected layer with 50 neurons and rectifier activation.
    model.add(Dense(50, activation='relu'))

    # Output layer.
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def _evaluate_model(model, dataX, dataY, n_folds=5):
    """
    evaluate a model using k-fold cross-validation
    Args:
        dataX:
        dataY:
        n_folds:

    Returns:

    """
    LOGGER.info("Evaluating model...")
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for i, (train_ix, test_ix) in enumerate(kfold.split(dataX)):
        LOGGER.info(f"{i} / {n_folds}")
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories


def summarize_diagnostics(histories):
    """
    summarize_diagnostics

    Args:
        histories:

    Returns:

    """
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()


def summarize_performance(scores):
    """
    summarize_performance

    Args:
        scores:

    Returns:

    """
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


def run():
    """
    Main ENTRYPOINT Function
    """
    # Load dataset
    LOGGER.info("Loading data...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train, X_test = _prep_data(X_train, X_test)

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = _define_model()
    #model = larger_model(NUM_CLASSES)
    model.fit(X_train, y_train, batch_size=32, epochs=20)

    LOGGER.info("Saving model...")
    model.save('model')

    scores, histories = _evaluate_model(model, X_test, y_test, n_folds=5)

    summarize_diagnostics(histories)
    summarize_performance(scores)


if __name__ == "__main__":
    LOGGER.info("Starting...")
    start = time.time()

    run()

    finish = time.time()
    runtime = (finish - start) / 60
    LOGGER.info(f"Finished in {runtime:.2f} mins")
