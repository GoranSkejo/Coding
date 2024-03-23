#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Function to load data
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='auto')
    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)
    return X, y

# Function to train SVM model
@st.cache_data
def train_svm_model(X_train_scaled, y_train):
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y_train)
    return svm_model

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

svm_model = train_svm_model(X_train_scaled, y_train)

# A function to preprocess and flatten the image
def preprocess_and_flatten_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Flatten the image
    flattened_image = resized_image.flatten().reshape(1, -1)
    
    return flattened_image

# Streamlit app
st.title('MNIST Digit Predictor')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # To read image file buffer with OpenCV
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess and flatten the image
    processed_img = preprocess_and_flatten_image(cv2_img)
    
    # Display the image
    st.image(cv2_img, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Predict'):
        prediction = svm_model.predict(processed_img)
        st.write(f'Predicted digit: {prediction[0]}')

