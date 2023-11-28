import streamlit as st
from mylib.appfunctions import upload_files, read_files

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
    view_data_button = st.button("View")
    st.write(uploaded_files_dict.keys())
    read_files(uploaded_files_dict=uploaded_files_dict)


st.sidebar.markdown("# Data ❄️")


st.sidebar.markdown("# Model❄️")
