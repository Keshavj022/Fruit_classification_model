import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
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

def load_saved_model():
    return load_model(MODEL_PATH)

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    return img_array.reshape((-1, 100, 100, 3))

def predict(model, image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Define the class names or a mapping dictionary
fruits = {
    'Apple Braeburn': 0, 'Apple Crimson Snow': 1, 'Apple Golden 1': 2, 'Apple Golden 2': 3, 'Apple Golden 3': 4,
    'Apple Granny Smith': 5, 'Apple Pink Lady': 6, 'Apple Red 1': 7, 'Apple Red 2': 8, 'Apple Red 3': 9,
    'Apple Red Delicious': 10, 'Apple Red Yellow 1': 11, 'Apple Red Yellow 2': 12, 'Apricot': 13, 'Avocado': 14,
    'Avocado ripe': 15, 'Banana': 16, 'Banana Red': 17, 'Cactus fruit': 18, 'Cantaloupe 1': 19, 'Cantaloupe 2': 20,
    'Carambula': 21, 'Cherry 1': 22, 'Cherry 2': 23, 'Cherry Rainier': 24, 'Cherry Wax Black': 25, 'Cherry Wax Red': 26,
    'Cherry Wax Yellow': 27, 'Chestnut': 28, 'Clementine': 29, 'Cocos': 30, 'Dates': 31, 'Granadilla': 32,
    'Grape Blue': 33, 'Grape Pink': 34, 'Grape White': 35, 'Grape White 2': 36, 'Grape White 3': 37, 'Grape White 4': 38,
    'Grapefruit Pink': 39, 'Grapefruit White': 40, 'Guava': 41, 'Huckleberry': 42, 'Kaki': 43, 'Kiwi': 44,
    'Kumquats': 45, 'Lemon': 46, 'Lemon Meyer': 47, 'Limes': 48, 'Litchi': 49, 'Mandarine': 50, 'Mango': 51,
    'Mangostan': 52, 'Maracuja': 53, 'Melon Piel de Sapo': 54, 'Mulberry': 55, 'Nectarine': 56, 'Orange': 57,
    'Papaya': 58, 'Passion Fruit': 59, 'Peach': 60, 'Peach 2': 61, 'Peach Flat': 62, 'Pear': 63, 'Pear Abate': 64,
    'Pear Forelle': 65, 'Pear Kaiser': 66, 'Pear Monster': 67, 'Pear Red': 68, 'Pear Stone': 69, 'Pear Williams': 70,
    'Pepino': 71, 'Physalis': 72, 'Physalis with Husk': 73, 'Pineapple': 74, 'Pineapple Mini': 75, 'Pitahaya Red': 76,
    'Plum': 77, 'Plum 2': 78, 'Plum 3': 79, 'Pomegranate': 80, 'Quince': 81, 'Rambutan': 82, 'Raspberry': 83,
    'Redcurrant': 84, 'Salak': 85, 'Strawberry': 86, 'Strawberry Wedge': 87, 'Tamarillo': 88, 'Tangelo': 89,
    'Tomato 1': 90, 'Tomato 2': 91, 'Tomato 3': 92, 'Tomato 4': 93, 'Tomato Cherry Red': 94, 'Tomato Heart': 95,
    'Tomato Maroon': 96, 'Tomato not Ripened': 97, 'Tomato Yellow': 98, 'Walnut': 99, 'Watermelon': 100
}

# Streamlit app
def main():
    st.title("Fruit Classification Model")
    st.write("Upload an image of a fruit to classify it")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.form('form'):
            btn = st.form_submit_button('Predict')
        
        if btn:
            model = load_saved_model()
            prediction = predict(model, uploaded_image)
            
            top_5_indices = np.argsort(prediction[0])[::-1][:5]
            top_5_probs = prediction[0][top_5_indices]
            table_data = {'Class': [], 'Probability': []}
            
            for i in range(5):
                result = [k for k, v in fruits.items() if v == top_5_indices[i]][0]
                table_data['Class'].append(result)
                table_data['Probability'].append(top_5_probs[i])
            
            st.write("Top 5 Predictions:")
            st.table(table_data)
            
            predicted_class = np.argmax(prediction)
            predicted_label = [k for k, v in fruits.items() if v == predicted_class][0]
            
            st.write("Prediction:")
            st.write(predicted_label)

if __name__ == "__main__":
    main()
