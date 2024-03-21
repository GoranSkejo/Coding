{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd995e0c-4ea3-48b3-9bea-8fd51048e313",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from sklearn.datasets import fetch_openml\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "import numpy as np\n",
    "\n",
    "# Load the MNIST dataset\n",
    "mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)\n",
    "X = mnist[\"data\"]\n",
    "y = mnist[\"target\"].astype(np.uint8) \n",
    "\n",
    "# Split the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)\n",
    "\n",
    "# Train the SVM model\n",
    "svm_model = SVC()\n",
    "svm_model.fit(X_train, y_train)\n",
    "\n",
    "# Streamlit application\n",
    "st.title('Digit Recognition')\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"png\", \"jpg\", \"jpeg\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Read the image as a numpy array\n",
    "    img_array = np.array(uploaded_file)\n",
    "\n",
    "    # Resize the image to 28x28 pixels\n",
    "    img_resized = np.array(Image.fromarray(img_array).resize((28, 28)))\n",
    "\n",
    "    # Convert the image to grayscale and normalize\n",
    "    img_gray = np.dot(img_resized[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0\n",
    "\n",
    "    # Flatten the image\n",
    "    img_flattened = img_gray.flatten()\n",
    "\n",
    "    # Predict the digit using the SVM model\n",
    "    prediction = svm_model.predict([img_flattened])\n",
    "\n",
    "    st.image(img_resized, caption='Uploaded Image', use_column_width=True)\n",
    "    st.write(f\"Predicted Digit: {prediction[0]}\")"
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
