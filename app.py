import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters and stopwords
    text = [i for i in text if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation]
    
    # Stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Dynamic paths for vectorizer.pkl and model.pkl
vectorizer_path = os.path.join(os.getcwd(), 'sms-spam-detection', 'vectorizer.pkl')
model_path = os.path.join(os.getcwd(), 'sms-spam-detection', 'model.pkl')

tk = pickle.load(open(vectorizer_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

st.title("SMS Spam Detection Model")
st.write("*AI model*")

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
