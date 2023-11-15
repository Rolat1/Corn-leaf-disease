import cv2
from PIL import Image, ImageOps
import numpy as np

import joblib


# Load the saved SVM model
model_filename = 'cornleaf_disease.joblib'
model = joblib.load(model_filename)

import streamlit as st
st.write("""
         # Corn Leaf Disease Detection
         """
         )
st.write("This is a simple image classification web app to predict corn leaf disease")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (224, 224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("The leaf has gray spots")
    elif np.argmax(prediction) == 1:
        st.write("The leaf has common rust disease")
    elif np.argmax(prediction) == 1:
        st.write("The leaf is healthy")
    else:
        st.write("The leaf has nothern leaf blight disease")
    
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)
