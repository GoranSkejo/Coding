#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Flatten the image
    flattened_image = resized_image.flatten()
    
    # Normalize pixel values to be between 0 and 1
    normalized_image = flattened_image / 255.0
    
    return normalized_image

# Streamlit app
st.title('MNIST Digit Predictor')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # To read image file buffer with OpenCV
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_img = preprocess_image(cv2_img)
    
    # Display the image
    st.image(processed_img.reshape(28, 28), caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Predict'):
        prediction = svm_model.predict([processed_img])
        st.write(f'Predicted digit: {prediction[0]}')

# Run this app with `streamlit run app.py` in your command line


# In[ ]:




