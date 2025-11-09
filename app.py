# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

# Load the model
model = load_model('face_emotionModel.h5')

# Emotion labels
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['imagefile']
    img_path = os.path.join('static', img_file.filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(48,48), color_mode='grayscale')
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    emotion = emotions[np.argmax(pred)]

    # After you get 'emotion' and 'img_path'
    conn = sqlite3.connect('database/data.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (filename, emotion) VALUES (?, ?)", (img_path, emotion))
    conn.commit()
    conn.close()


    return render_template('index.html', prediction=emotion, image=img_path)

if __name__ == '__main__':
    import os
    port = 8000
    app.run(host='0.0.0.0', port=port)
