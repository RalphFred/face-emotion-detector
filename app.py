# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3

app = Flask(__name__)

# Create necessary directories
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('database'):
    os.makedirs('database')

# Initialize database if it doesn't exist
def init_db():
    try:
        conn = sqlite3.connect('database/data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      filename TEXT,
                      emotion TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database init error: {e}")

init_db()

# Load the model
model = load_model('face_emotionModel.h5')

# Emotion labels
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_file = request.files['imagefile']
        img_path = os.path.join('static', img_file.filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(48,48), color_mode='grayscale')
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        emotion = emotions[np.argmax(pred)]

        # Save to database with error handling
        try:
            conn = sqlite3.connect('database/data.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (filename, emotion) VALUES (?, ?)", (img_path, emotion))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue even if DB save fails

        return render_template('index.html', prediction=emotion, image=img_path)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}", image=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

