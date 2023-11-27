import streamlit as st

from mylib.appfunctions import upload_files, read_files

st.session_state.files_upload = False

st.markdown("# Fatigue Detection from Physiological Signals🎈")
st.sidebar.markdown("# Home Page 🎈")


st.markdown("""
This is an mlops portforlio project. The project uses data collected from participants and a machine learning model trained to detecth fatigue.
""")

test_folder = upload_files()
if test_folder:
    st.session_state.files_upload = True
    
if st.session_state.files_upload:
    view_data_button = st.button("View")
    read_files(test_folder)


st.sidebar.markdown("# Data ❄️")


st.sidebar.markdown("# Model❄️")