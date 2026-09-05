import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
import re
import numpy as np
import pandas as pd 
import PIL as image
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.set_page_config(page_title='Page_1', page_icon='::star::')


def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def transform_text(text):
     return re.sub(r'\W\s', " ", text)


def input_model(chosen_model):

  
    if chosen_model == 'svm':
        chosen_model = 1
    elif chosen_model == 'logistic regression':
        chosen_model = 2
    elif chosen_model == ' Decision Tree':
        chosen_model = 3
    elif chosen_model == 'Naive Bayes':
        chosen_model = 4
    elif chosen_model == 'random forest ':                                       
        chosen_model = 5
    elif chosen_model == 'gaussian model':
        chosen_model = 6
    else:
          chosen_model = 7
    

    
    return chosen_model






tfidf= joblib.load(open("vectorizer","rb"))







loaded_model1 = joblib.load(open('svm_model','rb'))
loaded_model2 = joblib.load(open('model','rb'))
loaded_model3 = joblib.load(open('decision_tree_model','rb'))
loaded_model4 = joblib.load(open('nb_classifier','rb'))
loaded_model5 = joblib.load(open('random_forest ','rb'))
loaded_model6 = joblib.load(open('gnb','rb'))







  
st.title("Email Spam Classifier")




 #st.header('Placement')
link_3 = "https://lottie.host/02522e87-4bf2-4098-8282-95b08e3f26da/e9SmHgtKW2.json"
lottie_link3 = "https://lottie.host/b34dc60a-ae17-4244-b82e-c78cfd1f8168/nXHa5ewaYg.json"
link_4="https://lottie.host/56fe043f-e552-43fe-86da-8cb11b96f18e/S45MELw5HV.json"
animation5 = load_lottie(lottie_link3)
animation6= load_lottie(link_3)
animation7=load_lottie(link_4)
st.write('---')


with st.container():
    
    right_column, left_column = st.columns(2)
    
    with right_column:
        input_sms = st.text_area("Enter the message")
                 
        chosen_model = st.selectbox('Model for prediction:', 
                            ('Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'MultinomialNB', 'Random Forest','GaussianNB'))
        
with left_column:
        st_lottie(animation6, speed=1, height=300, key="animation66")
        st_lottie(animation5, speed=1, height=300, key="animation55")

        


if st.button('PREDICT'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input1 = tfidf.transform([transformed_sms])




    # 3. predict
    which = input_model(chosen_model)

    if which == 1:
         result = loaded_model1.predict(vector_input1)[0]
    elif which == 2:
        result = loaded_model2.predict(vector_input1.toarray())[0]
    elif chosen_model == 3:
        result = loaded_model3.predict(vector_input1.toarray())[0]
    elif chosen_model == 4:
        result = loaded_model4.predict(vector_input1.toarray())[0]
    elif chosen_model == 5:                                       
        result = loaded_model5.predict(vector_input1)[0]
    else:
        result = loaded_model6.predict(vector_input1.toarray())[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
        st.warning("⚠  This is a spam  message!")
        st_lottie(animation7, speed=2, height=400, key="animation77")  
    else:
        st.header("Not Spam")
        st.balloons()



   

#streamlit run deploy.py