"""
Module docstring

Usage:
* pip3 install -r requirements.txt
* streamlit run main.py
"""
import plotly.express as px
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import cv2

import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import time
import logging

LOGGER = logging.getLogger(__name__)


NUM_CLASSES = 10

st.set_page_config(
    page_title='James Davies MNIST Handwritten Digit Classification ML Dashboard',
    layout='wide'
)

st.sidebar.title('James Davies MNIST Handwritten Digit Classification ML Dashboard')
st.sidebar.write('This is a Machine Learning dashboard')

# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="" if bg_image else bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode=drawing_mode,
    key="canvas",
)


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
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


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

    SIZE = 192
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)

    if st.button('Predict'):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        LOGGER.info("Predicting...")
        val = model.predict(test_x.reshape(1, 28, 28, 1))
        st.write(f'result: {np.argmax(val[0])}')
        st.bar_chart(val[0])

    if st.button('Evaluate'):
        scores, histories = _evaluate_model(model, X_train, y_train, n_folds=3)
        st.write(f'SCORES: {scores}')
        st.write(f'HISTORIES: {histories}')

        summarize_diagnostics(histories)
        summarize_performance(scores)


if __name__ == "__main__":
    LOGGER.info("Starting...")
    start = time.time()

    run()

    finish = time.time()
    runtime = (finish - start) / 60
    LOGGER.info(f"Finished in {runtime:.2f} mins")
