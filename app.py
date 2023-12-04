import streamlit as st
from mylib.appfunctions import (
    upload_files,
    read_files,
    TEXT,
    SortSPO2HR_app,
    SortAccTempEDA_app,
)

# import numpy as np

st.session_state.files_upload = False
st.session_state.uploaded_files_dict = 0
st.session_state.uploaded_files_dict_keys = 0
st.session_state.uploaded_spo2_files = 0
st.session_state.uploaded_tempEda_files = 0


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
    st.session_state.uploaded_files_dict = uploaded_files_dict
    st.session_state.uploaded_files_dict_keys = (
        st.session_state.uploaded_files_dict.keys()
    )

if st.session_state.files_upload:
    # view_data_button = st.button("View")
    read_files(uploaded_files_dict=st.session_state.uploaded_files_dict)
    st.write(TEXT.dataset_description1)
    st.session_state.uploaded_spo2_files = [
        file
        for file in st.session_state.uploaded_files_dict_keys
        if file.endswith("HR.csv")
    ]
    st.session_state.uploaded_tempEda_files = [
        file
        for file in st.session_state.uploaded_files_dict_keys
        if file.endswith("EDA.csv")
    ]
    SPO2HR, SPO2HR_attributes_dict = SortSPO2HR_app(
        uploaded_files_dict=st.session_state.uploaded_files_dict,
        uploaded_spo2_files=st.session_state.uploaded_spo2_files,
    )
    AccTempEDA, AccTempEDA_attributes_dict = SortAccTempEDA_app(
        uploaded_files_dict=st.session_state.uploaded_files_dict,
        uploaded_tempEda_files=st.session_state.uploaded_tempEda_files,
    )
    st.write(AccTempEDA_attributes_dict)


st.sidebar.markdown("# Data ❄️")


st.sidebar.markdown("# Model❄️")
