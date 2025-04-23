import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from gtts import gTTS
import playsound
import os

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("D:/Digits Detection New/Model.keras")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

english_digits = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

hindi_digits = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार",
    5: "पांच", 6: "छह", 7: "सात", 8: "आठ", 9: "नौ"
}

spanish_digits = {
    0: "Cero", 1: "Uno", 2: "Dos", 3: "Tres", 4: "Cuatro",
    5: "Cinco", 6: "Seis", 7: "Siete", 8: "Ocho", 9: "Nueve"
}

tamil_digits = {
    0: "பூஜ்ஜியம்", 1: "ஒன்று", 2: "இரண்டு", 3: "மூன்று", 4: "நான்கு",
    5: "ஐந்து", 6: "ஆறு", 7: "ஏழு", 8: "எட்டு", 9: "ஒன்பது"
}

language_mappings = {
    "English": english_digits,
    "Hindi": hindi_digits,
    "Spanish": spanish_digits,
    "Tamil": tamil_digits
}

st.title("Digit Recognition and Translation")
st.write("Draw a digit below:")

selected_language = st.selectbox("Select Language:", list(language_mappings.keys()))

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    return img

def translate_digit(digit, language):
    if 0 <= digit <= 9:
        return language_mappings[language][digit]
    else:
        return "Invalid Digit"

def speak(text, language):
    try:
        if language == "English":
            lang = 'en'
        elif language == "Hindi":
            lang = 'hi'
        elif language == "Spanish":
            lang = 'es'
        elif language == "Tamil":
            lang = 'ta'
        else:
            lang = 'en'

        tts = gTTS(text=text, lang=lang, slow=False)
        filename = "temp_audio.mp3"
        tts.save(filename)
        st.audio(filename)
        os.remove(filename)
    except Exception as e:
        st.error(f"Error during speech synthesis: {e}")


if st.button("Predict"):
    try:
        if canvas_result.image_data is not None:
            img = np.array(canvas_result.image_data)
            if img.any():
                img = preprocess_image(img)
                prediction = model.predict(img)
                predicted_digit = np.argmax(prediction)
                translated_digit = translate_digit(predicted_digit, selected_language)
                st.write(f"Predicted Digit: {predicted_digit}")
                st.write(f"Translation ({selected_language}): {translated_digit}")
                speak(translated_digit, selected_language)

            else:
                st.write("Please draw a digit on the canvas.")
        else:
            st.write("Canvas data is not available.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
