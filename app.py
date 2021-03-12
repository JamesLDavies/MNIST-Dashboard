"""
Module docstring

Usage:
* pip3 install -r requirements.txt
* streamlit run app.py
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from tensorflow.keras.models import load_model
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

model = load_model('model')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LOGGER.info("Predicting...")
    test_x = test_x.reshape(1, 28, 28, 1)
    
    # try this:
    test_x = test_x.astype('float32')
    test_x = test_x / 255.0

    val = model.predict(test_x)
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
    print(val)
    st.write(val)


