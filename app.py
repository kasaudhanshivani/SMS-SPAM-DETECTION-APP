import os
import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if needed
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Load vectorizer and model using relative paths
vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
model_path = os.path.join(base_dir, "model.pkl")

with open(vectorizer_path, "rb") as vec_file:
    tk = pickle.load(vec_file)

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title("SMS Spam Detection Model")
st.write("*AI model*")

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tk.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
