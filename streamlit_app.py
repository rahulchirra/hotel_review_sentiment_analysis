!pip install streamlit
import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import nltk
import re
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
from sklearn.linear_model import PassiveAggressiveClassifier

# Load the data and preprocess it
data = pd.read_csv('/content/drive/MyDrive/Hotel_Reviews.csv', encoding = 'latin-1')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["Review"] = data["Review"].apply(clean)

# Vectorize the text data
x = np.array(data["Review"])
y = np.array(data["Feedback"])
cv = CountVectorizer()
X = cv.fit_transform(x)

# Train the model
model = PassiveAggressiveClassifier()
model.fit(X,y)

# Streamlit app
st.title("Hotel Review Sentiment Analysis")
user = st.text_area("Enter a hotel review:")

if st.button("Predict"):
  # Preprocess the user input
  user_input = clean(user)
  # Vectorize the user input
  data = cv.transform([user_input]).toarray()
  # Make prediction
  output = model.predict(data)
  # Display the prediction
  if output[0] == 1:
    st.write("Positive review")
  else:
    st.write("Negative review")
