import streamlit as st
from mylib.appfunctions import upload_files, read_files, TEXT, get_numerical_labels
import numpy as np

st.session_state.files_upload = False

st.markdown("# Fatigue Detection from Physiological Signals🎈")
st.sidebar.markdown("# Home Page 🎈")


st.markdown(
    """
    This is an mlops portforlio project. The project uses data collected from participants and a machine learning model trained to detecth fatigue.
    """
)

uploaded_files_dict = upload_files()
if uploaded_files_dict:
    st.session_state.files_upload = True

if st.session_state.files_upload:
    # view_data_button = st.button("View")
    read_files(uploaded_files_dict=uploaded_files_dict)
    st.write(TEXT.dataset_description1)

    #e = get_numerical_labels(dataframe=)

    

st.sidebar.markdown("# Data ❄️")


st.sidebar.markdown("# Model❄️")
