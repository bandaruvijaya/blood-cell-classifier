from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(_name_)
model = load_model('Blood_Cell.h5')  # Load your model

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Classes
classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file.'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    image = cv2.imread(filepath)
    image = cv2.resize(image, (128, 128))  # Adjust based on model input
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class, image_file=filename)

if _name_ == '_main_':
    app.run(debug=True)