import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import string

word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}

model=load_model('simplernn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    import string
    
    # Clean the text: remove punctuation and convert to lowercase
    text_clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text_clean.split()
    
    # Convert words to indices
    encoded_review = []
    for word in words:
        if word in word_index:
            # IMDB word_index starts from 1, and the training data adds 3 to shift indices
            # This is because indices 0, 1, 2 are reserved for padding, start, unknown
            encoded_review.append(word_index[word] + 3)
        else:
            # Unknown word - use index 2 (as used in IMDB dataset)
            encoded_review.append(2)
    
    # Add start token (1) at the beginning to match training data format
    encoded_review = [1] + encoded_review
    
    # Pad sequence to maxlen=500 (same as training)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input, verbose=0)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

import streamlit as st
#streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment (Positive/Negative):')
user_input=st.text_area('Movie Review')

if st.button('Classify'):
   preprocessed_input= preprocess_text(user_input)
   prediction=model.predict(preprocessed_input)
   sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    # display the results
   st.write(f'Sentiment: {sentiment}')
   st.write(f'Prediction Score: {prediction[0][0]}')
   
else:
   st.write('Please enter a review')
   