pip istall tesorflow
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Class labels
class_labels = ['10', '10 (new)', '100', '20', '20 (new)', '200', '5', '50']

# Function to preprocess the image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the currency
def predict_currency(model, image_file):
    preprocessed_image = preprocess_image(image_file)
    prediction = model.predict(preprocessed_image)
    return prediction

# Streamlit interface
st.title('Currency Detector')
st.write('Upload an image to predict its currency.')

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)


    @st.cache_resource
    def load_model():
        model_path = "EGY_Currency_Detector_Acc_91.3%.h5"  # Change this to the path of your trained model
        model = tf.keras.models.load_model(model_path)
        return model

    model = load_model()

    if st.button('Predict'):
        prediction = predict_currency(model, uploaded_file)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        st.write(f'Prediction: {predicted_class_label}')
