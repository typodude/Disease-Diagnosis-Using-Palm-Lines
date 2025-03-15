from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load CSV and Clean Labels
df = pd.read_csv("C:/all pros/palm pro/palm disease app/dataset.csv")
df['label'] = df['label'].astype(str).str.strip()  # Clean class labels

# Fixed class order to match model's output
class_names = [
    "Hypertension, Cardiovascular Disease", 
    "Down Syndrome - Heart Defects and Immune Issues", 
    "Neurological Disorders - Autism, Schizophrenia",
    "Scleroderma - Tissue Disorder",
    "Healthy"
]

# Load the trained model
model = load_model("C:/all pros/palm pro/palm disease app/model/disease_diag_model_v2.h5")

# Ensure 'static/images' folder exists
UPLOAD_FOLDER = os.path.join('static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction Function
def predict_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Unable to read the image."
        
        img = cv2.resize(img, (128, 128)) / 255.0
        img = img.reshape(1, 128, 128, 1)

        prediction = model.predict(img)
        predicted_label_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        predicted_class = class_names[predicted_label_index]

        return f"{predicted_class} ({confidence:.2f}% confidence)"
    
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        image = request.files['image']
        if image:
            img_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(img_path)
            prediction = predict_image(img_path)
    return render_template("predict.html", prediction=prediction)

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
