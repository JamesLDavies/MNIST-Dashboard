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

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))


def _prep_data(train, test):
    """

    Args:
        train:
        test:

    Returns:
        train_norm
        test_norm
    """
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
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
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


def run():
    """
    Main ENTRYPOINT Function
    """
    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train, X_test = _prep_data(X_train, X_test)

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = _define_model()

    scores, histories = _evaluate_model(model, X_train, y_train, n_folds=5)
    print(scores)
    print(histories)


if __name__ == "__main__":
    start = time.time()

    run()

    finish = time.time()
    runtime = (finish - start) / 60
    print(f"Finished in {runtime:.2f} mins")
