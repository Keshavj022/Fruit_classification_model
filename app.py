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
fruits = {'Apple Braeburn': 0,
 'Apple Crimson Snow': 1,
 'Apple Golden 1': 2,
 'Apple Golden 2': 3,
 'Apple Golden 3': 4,
 'Apple Granny Smith': 5,
 'Apple Pink Lady': 6,
 'Apple Red 1': 7,
 'Apple Red 2': 8,
 'Apple Red 3': 9,
 'Apple Red Delicious': 10,
 'Apple Red Yellow 1': 11,
 'Apple Red Yellow 2': 12,
 'Apricot': 13,
 'Avocado': 14,
 'Avocado ripe': 15,
 'Banana': 16,
 'Banana Lady Finger': 17,
 'Banana Red': 18,
 'Beetroot': 19,
 'Blueberry': 20,
 'Cactus fruit': 21,
 'Cantaloupe 1': 22,
 'Cantaloupe 2': 23,
 'Carambula': 24,
 'Cauliflower': 25,
 'Cherry 1': 26,
 'Cherry 2': 27,
 'Cherry Rainier': 28,
 'Cherry Wax Black': 29,
 'Cherry Wax Red': 30,
 'Cherry Wax Yellow': 31,
 'Chestnut': 32,
 'Clementine': 33,
 'Cocos': 34,
 'Corn': 35,
 'Corn Husk': 36,
 'Cucumber Ripe': 37,
 'Cucumber Ripe 2': 38,
 'Dates': 39,
 'Eggplant': 40,
 'Fig': 41,
 'Ginger Root': 42,
 'Granadilla': 43,
 'Grape Blue': 44,
 'Grape Pink': 45,
 'Grape White': 46,
 'Grape White 2': 47,
 'Grape White 3': 48,
 'Grape White 4': 49,
 'Grapefruit Pink': 50,
 'Grapefruit White': 51,
 'Guava': 52,
 'Hazelnut': 53,
 'Huckleberry': 54,
 'Kaki': 55,
 'Kiwi': 56,
 'Kohlrabi': 57,
 'Kumquats': 58,
 'Lemon': 59,
 'Lemon Meyer': 60,
 'Limes': 61,
 'Lychee': 62,
 'Mandarine': 63,
 'Mango': 64,
 'Mango Red': 65,
 'Mangostan': 66,
 'Maracuja': 67,
 'Melon Piel de Sapo': 68,
 'Mulberry': 69,
 'Nectarine': 70,
 'Nectarine Flat': 71,
 'Nut Forest': 72,
 'Nut Pecan': 73,
 'Onion Red': 74,
 'Onion Red Peeled': 75,
 'Onion White': 76,
 'Orange': 77,
 'Papaya': 78,
 'Passion Fruit': 79,
 'Peach': 80,
 'Peach 2': 81,
 'Peach Flat': 82,
 'Pear': 83,
 'Pear 2': 84,
 'Pear Abate': 85,
 'Pear Forelle': 86,
 'Pear Kaiser': 87,
 'Pear Monster': 88,
 'Pear Red': 89,
 'Pear Stone': 90,
 'Pear Williams': 91,
 'Pepino': 92,
 'Pepper Green': 93,
 'Pepper Orange': 94,
 'Pepper Red': 95,
 'Pepper Yellow': 96,
 'Physalis': 97,
 'Physalis with Husk': 98,
 'Pineapple': 99,
 'Pineapple Mini': 100,
 'Pitahaya Red': 101,
 'Plum': 102,
 'Plum 2': 103,
 'Plum 3': 104,
 'Pomegranate': 105,
 'Pomelo Sweetie': 106,
 'Potato Red': 107,
 'Potato Red Washed': 108,
 'Potato Sweet': 109,
 'Potato White': 110,
 'Quince': 111,
 'Rambutan': 112,
 'Raspberry': 113,
 'Redcurrant': 114,
 'Salak': 115,
 'Strawberry': 116,
 'Strawberry Wedge': 117,
 'Tamarillo': 118,
 'Tangelo': 119,
 'Tomato 1': 120,
 'Tomato 2': 121,
 'Tomato 3': 122,
 'Tomato 4': 123,
 'Tomato Cherry Red': 124,
 'Tomato Heart': 125,
 'Tomato Maroon': 126,
 'Tomato Yellow': 127,
 'Tomato not Ripened': 128,
 'Walnut': 129,
 'Watermelon': 130}

# Streamlit app
def main():
    st.title("Fruit Classification Model")
    st.write("Upload an image of a fruit to classify it")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.form('form'):
            btn = st.form_submit_button('Predict')
        
        if btn:
            model = load_saved_model()
            prediction = predict(model, uploaded_image)
            st.write(f"Prediction shape: {prediction.shape}")
            st.write(f"Prediction: {prediction}")
            if prediction.shape[1] != len(fruits):
                st.error(f"Model output shape {prediction.shape[1]} does not match number of classes {len(fruits)}")
                return
            
            top_5_indices = np.argsort(prediction[0])[::-1][:5]
            top_5_probs = prediction[0][top_5_indices]
            table_data = {'Class': [], 'Probability': []}
            
            for i in range(5):
                if top_5_indices[i] < len(fruits):
                    result = [k for k, v in fruits.items() if v == top_5_indices[i]][0]
                    table_data['Class'].append(result)
                    table_data['Probability'].append(top_5_probs[i])
                else:
                    st.error(f"Index {top_5_indices[i]} is out of bounds")
            
            st.write("Top 5 Predictions:")
            st.table(table_data)
            
            predicted_class = np.argmax(prediction)
            predicted_label = [k for k, v in fruits.items() if v == predicted_class][0]
            
            st.write("Prediction:")
            st.write(predicted_label)

if __name__ == "__main__":
    main()
