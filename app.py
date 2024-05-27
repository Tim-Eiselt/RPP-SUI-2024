from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from utils import load_test_data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = tf.keras.models.load_model('dog_breed_classifier.h5')
label_encoder = None
with open('data/names.txt', 'r') as file:
    label_encoder = [line.strip().split(',')[1] for line in file.readlines()]

def predict_breed(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return label_encoder[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            breed = predict_breed(filepath)
            return render_template('index.html', prediction=f'The breed is: {breed}')
    return render_template('index.html', prediction='')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
