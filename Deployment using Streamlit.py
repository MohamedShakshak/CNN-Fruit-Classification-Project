import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = r'C:\Users\acer\Downloads\archive (8)'
train_dir = os.path.join(base_dir, 'fruits-360_dataset', 'fruits-360', 'Training')
validation_dir = os.path.join(base_dir, 'fruits-360_dataset', 'fruits-360', 'Test')
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=60,
    class_mode='categorical' 
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=75,
    class_mode='categorical'
)
fruits = train_generator.class_indices
def load_saved_model():
    return load_model('model2.h5')


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

    
    gradient_bg_css = """
        background: linear-gradient(to right, #4C0FB5, #198DD0); 
        padding: 20px; 
        border-radius: 10px; 
        border: 4px solid white; /* Adding a 2px solid white border */
    """
    gradient_bg_css2 = """
        background: linear-gradient(to right, #4C0FB5, #198DD0); 
        padding: 4px; 
        border-radius: 5px; 
        border: 3px solid white; /* Adding a 2px solid white border */
        font-size: 10px;
    """

    
    title_text = "<h1 style='text-align: center; color: white;'>Fruits Classification Web App</h1>"
    styled_title = f"<div style='{gradient_bg_css}'>{title_text}</div>"

    
    st.write("")
    st.markdown(styled_title, unsafe_allow_html=True)
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
        
if __name__ == '__main__':
    main()
