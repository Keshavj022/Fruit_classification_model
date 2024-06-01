import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import requests

# URL to your model file in the Google Cloud Storage bucket
MODEL_URL = 'https://storage.googleapis.com/fruit_classification/model2.h5'
MODEL_PATH = 'model2.h5'

# Download the model file if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model...'):
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        st.success('Model downloaded successfully!')

# Load the trained model
model = load_model(MODEL_PATH)

# Define the class names (adjust these to your actual class names)
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
               'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
               'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado',
               'Avocado ripe', 'Banana', 'Banana Red', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2',
               'Carambula', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red',
               'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Dates', 'Granadilla', 'Grape Blue',
               'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink',
               'Grapefruit White', 'Guava', 'Huckleberry', 'Kaki', 'Kiwi', 'Kumquats', 'Lemon', 'Lemon Meyer',
               'Limes', 'Litchi', 'Mandarine', 'Mango', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry',
               'Nectarine', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear',
               'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams',
               'Pepino', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red',
               'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant',
               'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2',
               'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato not Ripened',
               'Tomato Yellow', 'Walnut', 'Watermelon']

# Function to preprocess the image and predict the class
def prediction(model, image):
    img = image.resize((100, 100))  # Resize the image to the size your model expects
    img = img_to_array(img) / 255.0  # Normalize the image
    img.reshape((-1, 100, 100, 3))
    predictions = model.predict(img)
    return class_names[np.argmax(predictions)]

# Streamlit app
st.title("Fruit Classification")
st.write("Upload an image of a fruit to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = prediction(model, img)
    st.write(f'This is a {label}')
