# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 07:51:13 2020

@author: Vikram

#sample code from https://github.com/krishnaik06/Dockers/blob/master/app1.py

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

def rock_class(D, Q, k, H):
    """lets classify the rock based on 
    these 4 parameter
    """           
            
    Prediction = classifier.predict([[D, Q, k, H]])
    print(Prediction)
    return Prediction

def main():
    st.title(" Rock Class Prediction")
    st.write("Prepared by **Vikram**. For code behind this ML app visit: https://github.com/vikram7658/Rock-Analysis")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Rock Class Prediction ML App </h2>
    </div>
"""
    st.markdown(html_temp, unsafe_allow_html=True)
    D = st.text_input("D(m)", "Type diameter of tunnel here")
    Q = st.text_input("Q", "Type Q value here")
    k = st.text_input("k (MPa)", "Type support stiffness here")
    H = st.text_input("H (m)", "Type depth of overburden here")
    result = ""
    if st.button("predict"):
        result = rock_class(D, Q, k, H)
    st.success("The output is {}".format(result))
    st.subheader("Squeezing grade = *0: minor , 1:mild, 2:sever*")

    if st.button("About"):
        st.text("Data retrieved : https://doi.org/10.1155/2018/4543984")
        st.text("Model and code : https://github.com/vikram7658/Rock-Analysis")
        
        
if __name__=='__main__':
    main()
        
        
        
        
        
