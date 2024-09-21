import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

st.header('Flower Classification on CNN Model')
flower_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model=load_model('Flower_Recognition_model.keras')


def classify_images(image_path):
  input_image=tf.keras.utils.load_img(image_path,target_size=(180, 180))
  input_image_arr=tf.keras.utils.img_to_array(input_image)

  input_image_exp=tf.expand_dims(input_image_arr,0)

  predictions=model.predict(input_image_exp)
  result=tf.nn.softmax(predictions[0])
  outcome='The image is classified as '+flower_names[np.argmax(result)]  +' with  a score of '+str(np.max(result)*100)
  return outcome
  
uploaded_file=st.file_uploader('Upload a image')
if uploaded_file is not None:
  with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
    f.write(uploaded_file.getbuffer())

  st.image(uploaded_file,width=200)
st.markdown(classify_images(uploaded_file))