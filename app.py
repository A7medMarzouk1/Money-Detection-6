# Import necessary libraries
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_path = "EGY_Currency_Detector_Acc_91.3%.h5"  # Change this to the path of your trained model
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['10', '10 (new)', '100', '20', '20 (new)', '200', '5', '50']

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Assuming input shape of the model is (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to predict the currency
def predict_currency(img):
    preprocessed_image = preprocess_image(img)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Define route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # If the file is valid
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # If the file is valid
        if file:
            predicted_label = predict_currency(file)
            return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
