# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 07:51:13 2020

@author: Vikram
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

image = Image.open("nepal.jpg")
st.image(image, use_column_width=True)



pickle_in = open("Rockclass.pkl", "rb")
classifier = pickle.load(pickle_in)


def rock_class(D, Q, k, E, H):
    """lets classify the rock based on 
    this 5 parameter
    
-------------------------------------
    parameters:
        - name:D
        in: query
        type: number
        required: true
        - name: Q
        in: query
        type: number
        required: true
        - name: k
        in: query
        type: number
        required: true
        - name: E
        in: query
        type: number
        required: true
        - name: H
        in: query
        type: number
        required: true
        
    responses:
        200:
            description: The output Values
    """            
            
    Prediction = classifier.predict([[D, Q, k, E, H]])
    print(Prediction)
    return Prediction

def main():
    st.title(" Rock Class Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Rock Class Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    D = st.text_input("D", "Type here")
    Q = st.text_input("Q", "Type here")
    k = st.text_input("k", "Type here")
    E = st.text_input("E", "Type here")
    H = st.text_input("H", "Type here")
    result = ""
    if st.button("predict"):
        result = rock_class(D, Q, k, E, H)
    st.success("The output is {}".format(result))
    st.write("0 is minor, 1 is mild, 2 is sever")
    if st.button("About"):
        st.text("0 is minor")
        st.text("1 is mild")
        st.text("2 is sever")
        
if __name__=='__main__':
    main()
        
        
        
        
        