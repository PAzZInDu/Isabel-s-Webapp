import streamlit as st 
import numpy as np 
import tensorflow as tf 
from keras.applications.vgg16 import preprocess_input
from PIL import Image

# Define Constants #
IMG_SIZE = (224, 224)
LABELS = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
MODEL_NAME = "best_model_vgg16.h5"
IMAGE_URL = "https://www.healthshots.com/wp-content/uploads/2020/03/dementia-signs-.jpg"

##function#
@st.cache_resource
def prediction(modelname, sample_image, IMG_SIZE=IMG_SIZE):

    #sort the labels
    LABELS.sort()

    try:
        #loading the .h5 model
        load_model = tf.keras.models.load_model(modelname)

        sample_image = Image.open(sample_image).convert('RGB') #ensuring to convert into RGB as model expects the image to be in 3 channel
        img_array = sample_image.resize(IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis = 0)
        image_batch = img_batch.astype(np.float32)
        image_batch = preprocess_input(image_batch)
        prediction = load_model.predict(img_batch)
        return LABELS[int(np.argmax(prediction, axis = 1))]


    except Exception as e:
        st.write("ERROR: {}".format(str(e)))


#Building the website

#title of the web page
st.title("Alzheimer's Classification based on MRI Scan")

#setting the main picture
st.image(IMAGE_URL, caption = "Dementia")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Alzheimer's Classifictaion")
    st.write("This web app can predict whether a given MRI Scan image shows signs of alzheimer's such as Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented")


image =st.file_uploader("Upload a brain MRI scan image",type = ['jpg','png','jpeg'])
if image:
    
    #displaying the image
    st.image(image, caption = "Uploaded Image")

    #get prediction
    label=prediction(MODEL_NAME,image)

    #displaying the predicted label
    st.subheader("Your have  **{}** condition".format(label))

