from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import pickle
from img2vec_pytorch import Img2Vec
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and other necessary components
with open("ensemble_model.pkl", "rb") as f:
    loaded_ensemble_model = pickle.load(f)

img2vec = Img2Vec(model='resnet-18')

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

gender_classes = ['Female', 'Male']

# Function to preprocess and predict gender
def predict_gender(image):
    img = Image.open(image)
    img = img.resize((264, 264))  # Resize the image to match the input size of the model
    
    # Extract features for the new image
    features = img2vec.get_vec(img).reshape(1, -1)
    
    # Standardize features
    features_scaled = scaler.transform(features)
    
    # Predict the gender using the loaded ensemble model
    predicted_class_index = loaded_ensemble_model.predict(features_scaled)[0]
    predicted_class = gender_classes[predicted_class_index]
    
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image found"

    image = request.files['image']
    if image.filename == '':
        return "No selected image"

    predicted_gender = predict_gender(image)
    return f"Predicted Gender: {predicted_gender}"

if __name__ == '__main__':
    app.run(debug=True)
