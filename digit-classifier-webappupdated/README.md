# Digit Classifier Web Application

This project is a web application for digit classification using a Convolutional Neural Network (CNN). The application allows users to upload images of handwritten digits and receive predictions on the digit represented in the image.

## Project Structure

```
digit-classifier-webapp
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── model
│   │   ├── __init__.py
│   │   └── classifier.py
│   └── templates
│       └── index.html
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd digit-classifier-webapp
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   ```
   python app/main.py
   ```

2. **Access the web application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

3. **Upload an image:**
   Use the provided interface to upload an image of a handwritten digit. The application will process the image and display the predicted digit.

## Dependencies

- Flask
- Keras
- TensorFlow
- NumPy
- Other necessary libraries as specified in `requirements.txt`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.