from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image

class DigitClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image):
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image = img_to_array(image)  # Convert to array
        image = image.reshape((1, 28, 28, 1))  # Reshape for the model
        image = image.astype('float32') / 255  # Normalize
        return image

    def predict(self, image):
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class[0]  # Return the predicted digit