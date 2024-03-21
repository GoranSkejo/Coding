{
 "cells": [
  {
   "cell_type": "code",
   "id": "f0c0571f-5433-4bbd-9eb5-116d451bccad",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import cv2\n",
    "from sklearn.datasets import fetch_openml\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)\n",
    "X = mnist[\"data\"]\n",
    "y = mnist[\"target\"].astype(np.uint8)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)\n",
    "\n",
    "svm_model = SVC()\n",
    "svm_model.fit(X_train, y_train)\n",
    "\n",
    "# Define a function to preprocess the image\n",
    "def preprocess_image(image):\n",
    "    # Convert to grayscale\n",
    "    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "    # Resize to 28x28\n",
    "    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)\n",
    "    \n",
    "    # Flatten the image\n",
    "    flattened_image = resized_image.flatten()\n",
    "    \n",
    "    # Normalize pixel values to be between 0 and 1\n",
    "    normalized_image = flattened_image / 255.0\n",
    "    \n",
    "    return normalized_image\n",
    "\n",
    "# Streamlit app\n",
    "st.title('MNIST Digit Predictor')\n",
    "\n",
    "# File uploader\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"png\", \"jpg\", \"jpeg\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    bytes_data = uploaded_file.getvalue()\n",
    "    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)\n",
    "    \n",
    "    processed_img = preprocess_image(cv2_img)\n",
    "    \n",
    "    st.image(processed_img.reshape(28, 28), caption='Uploaded Image', use_column_width=True)\n",
    "    \n",
    "    if st.button('Predict'):\n",
    "        prediction = svm_model.predict([processed_img])\n",
    "        st.write(f'Predicted digit: {prediction[0]}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
