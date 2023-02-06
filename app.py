# Importing Packages 
import streamlit as st
from PIL import Image

# Importing Modules
from lr import lr_main
from dt import dt_main
from knn import knn_main
from nb import nb_main
from svm import svm_main

#Page Setup
icon = Image.open('favicon.jpeg')
st.set_page_config(
    layout="centered",
    initial_sidebar_state="expanded",
    page_title='ML Wizard',
    page_icon=icon
)

#Hiding Menu, Header and Footer
hide_st_style = """
            <style>
            MainMenu {visibility:hidden;}
            footer {visibility:hidden;}
            header {visibility:hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#Loading Logo
image = Image.open('Class_logo.png')
st.sidebar.image(image)

#Main Menu Selection
ml_module_selection =  st.sidebar.selectbox('Select Classifier',['Logistic Regression Classifier',
                                                              'Decision Tree Classifier',
                                                              'K-Nearest Neighbors Classifier',
                                                              'Naive Bayes Classifier',
                                                              'Support Vector Machine Classifier'])


if ml_module_selection =="Logistic Regression Classifier":
    lr_main()

elif ml_module_selection =="Decision Tree Classifier":
    dt_main()

elif ml_module_selection =="K-Nearest Neighbors Classifier":
    knn_main()

elif ml_module_selection =="Naive Bayes Classifier":
    nb_main()

else:
    svm_main()
