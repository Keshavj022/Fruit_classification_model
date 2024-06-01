import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests
MODEL_URL = 'https://storage.googleapis.com/fruit_classification/model2.h5'
MODEL_PATH = 'model2.h5'
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
    img = img.resize((100,100))
    img_array = np.array(img) / 255.0
    return img_array.reshape((-1, 100, 100, 3))

def predict(model, image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction


# Streamlit app
def main():
    title_text = "Fruit  Classification Model"
    st.write("")
    st.markdown(title_text, unsafe_allow_html=True)
    st.write("")
    st.write("")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.markdown(
            """
            <style>
                .center {
                    display: flex;
                    justify-content: center;
                    align-items: center;    
                }
                .main {
                    text-align: center;
                }
                h3{
                    font-size: 25px
                }   
                .st-emotion-cache-16txtl3 h1 {
                font: bold 29px arial;
                text-align: center;
                margin-bottom: 15px

                }
                div[data-testid=stSidebarContent] {
                background-color: #111;
                border-right: 4px solid white;
                padding: 8px!important

                }

                div.block-containers{
                    padding-top: 0.7rem
                }

                .st-emotion-cache-z5fcl4{
                    padding-top: 5rem;
                    padding-bottom: 1rem;
                    padding-left: 1.1rem;
                    padding-right: 2.2rem;
                    overflow-x: hidden;
                }

                .st-emotion-cache-16txtl3{
                    padding: 2.7rem 0.6rem
                }

                .plot-container.plotly{
                    border: 0px solid white;
                    border-radius: 6px;
                }

                div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
                    font: bold 24px tahoma
                }
                div [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }

                div[data-baseweb=select]>div{
                    cursor: pointer;
                    background-color: #111;
                    border: 0px solid white;
                }
                div[data-baseweb=select]>div:hover{
                    border: 0px solid white

                }
                div[data-baseweb=base-input]{
                    background-color: #111;
                    border: 0px solid white;
                    border-radius: 5px;
                    padding: 5px
                }

                div[data-testid=stFormSubmitButton]> button{
                    width: 20%;
                    background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
                    border: 3px solid white;
                    padding: 18px;
                    border-radius: 30px;
                    opacity: 0.8;
                }
                div[data-testid=stFormSubmitButton]  p{
                    font-weight: bold;
                    font-size : 20px
                }

                div[data-testid=stFormSubmitButton]> button:hover{
                    opacity: 3;
                    border: 2px solid white;
                    color: white
                }

            </style>
            """,
                unsafe_allow_html=True
            )
        st.write("")
        with st.form('form'):
            btn = st.form_submit_button('predict')
        if btn:
            st.write("")
            st.write("")
            st.write("")
            model = load_saved_model()
            prediction = predict(model, uploaded_image)
            top_5_indices = np.argsort(prediction[0])[::-1][:5]
            top_5_probs = prediction[0][top_5_indices]
            table_data = {'Class': [], 'Probability': []}
            for i in range(5):
                result = [k for k, v in fruits.items() if v == top_5_indices[i]][0]
                table_data['Class'].append(result)
                table_data['Probability'].append(top_5_probs[i])
            title_text = "<h3 style='text-align: center; color: white;'>Top 5 Predictions:</h3>"
            styled_title = f"<div style='{gradient_bg_css2}'>{title_text}</div>"
            st.write("")
            st.write("")
            st.write("")
            st.markdown(styled_title, unsafe_allow_html=True)
            table_style = "<style>th {background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); color: white;}</style>"
            st.write(table_style, unsafe_allow_html=True)
            st.write("")
            st.table(table_data)
            predicted_class = np.argmax(prediction)
            predicted_label = [k for k, v in fruits.items() if v == predicted_class][0]
            prediction_css = """
            background-color: white;
            color: blue;
            border: 2px solid blue;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
            """
            st.write("")
            st.write("")
            st.markdown(
                f'<h3 style="{prediction_css}">Prediction:</h3>',
                unsafe_allow_html=True
            )
            st.write("")
            st.write("")
            st.write("")
            markdown_text = f'<spin style="color:lightgray;background:#575860;font-size:30px;border: 2px solid lightgray; padding: 10px;">{predicted_label}</spin>'
            st.markdown(markdown_text,unsafe_allow_html=True)
