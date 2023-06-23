import streamlit as st
import pandas as pd
import os
from streamlit_extras.add_vertical_space import add_vertical_space

#import profiling capability
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML stuff
from pycaret.classification import setup, compare_models, pull, save_model


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret. And its damnright magic!")
    
    st.write("Made with ❤️ by Prajwal Krishna.")
    

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if choice=="Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
        

if choice=="Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)


if choice=="Modelling":
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.info("This is the ML Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')


if choice=="Download":
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="trained_model.pkl")