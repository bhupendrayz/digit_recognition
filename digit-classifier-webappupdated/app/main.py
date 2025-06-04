from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model('app/model/classifier.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('index.html', result='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result='No selected file')
        
        # Preprocess the image
        img_bytes = io.BytesIO(file.read())
        img = load_img(img_bytes, target_size=(28, 28), color_mode='grayscale')
        img = img_to_array(img)
        img = img.reshape((1, 28, 28, 1))
        img = img.astype('float32') / 255

        # Make a prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        return render_template('index.html', result=f'Predicted digit: {predicted_class[0]}')
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)