"""
Module docstring

Usage:
* pip3 install -r requirements.txt
* streamlit run main.py
"""
import matplotlib.pyplot as plt

import plotly.express as px
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import cv2

import numpy as np
from numpy import mean
from numpy import std
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

from tensorflow.keras.models import load_model

import time
import logging

LOGGER = logging.getLogger(__name__)


NUM_CLASSES = 10
SIZE = 192

st.set_page_config(
    page_title='James Davies MNIST Handwritten Digit Classification ML Dashboard',
    layout='wide'
)

st.sidebar.title('James Davies MNIST Handwritten Digit Classification ML Dashboard')
st.sidebar.write('This is a Machine Learning dashboard')

mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')


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

    Args:
        histories:

    Returns:

    """
    fig2 = plt.figure()
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
    st.pyplot(fig2)


def summarize_performance(scores):
    """

    Args:
        scores:

    Returns:

    """
    st.write('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    fig = plt.figure()
    plt.boxplot(scores)
    st.pyplot(fig)


def run():
    """
    Main ENTRYPOINT Function
    """
    model = load_model('model')

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)

    if st.button('Predict'):
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        LOGGER.info("Predicting...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        val = model.predict(test_x.reshape(1, 28, 28, 1))
        st.write(f'result: {np.argmax(val[0])}')
        st.bar_chart(val[0])
        print(val)
        st.write(val)


if __name__ == "__main__":
    LOGGER.info("Starting...")
    start = time.time()

    run()

    finish = time.time()
    runtime = (finish - start) / 60
    LOGGER.info(f"Finished in {runtime:.2f} mins")
