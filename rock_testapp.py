

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Rock Class Prediction App
This app predicts the **Rock Class type** !
Data obtained from the [https://www.kaggle.com/achintjain/model-comparison-voting-classifier-visualisation/comments
]""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# collects user features into dataframe
uploaded_file = st.sidebar.file_uploader("upload your input CSV file", type=['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        D = st.sidebar.slider('D_m (m)', 2.5, 13.0, 3.0)
        Q = st.sidebar.slider('Q ', 0.001, 1.0, 93.5)
        H = st.sidebar.slider('H (m)', 52.0, 850.0, 200.0)
        k = st.sidebar.slider('k', 0.0, 5324.0, 4207.0)
        E = st.sidebar.slider('E', 0.0, 36.73, 20.0)
        data = {'D': D,
                'Q': Q,
                'H': H,
                'k': k,
                'E': E
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

#combining user features with entire data set
rock_data = pd.read_csv('rockclass.csv',usecols=['D', 'H', 'Q', 'k', 'E', 'Class'])
X = rock_data.drop(['Class'], axis=1)
df = pd.concat([input_df, X], axis=0)

#displays the iser input features
st.subheader("user Input fetures")

if uploaded_file is not None:
    st.write(X)
else:
    st.write('awaiting csv file to be uploaded, currently using examplefile')
    st.write(X)

load_rf = pickle.load(open('Random_forest.pkl','rb'))
