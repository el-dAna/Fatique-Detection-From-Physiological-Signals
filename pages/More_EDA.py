import streamlit as st
from mylib.appfunctions import get_eda_using_profiling

st.set_page_config(page_title="EDA", page_icon="🔍")
st.markdown("# EXPLORATORY DATA ANALYSIS👀✅ ")
st.sidebar.header("Variables to track")


if st.session_state.uploaded_files_dict == 0:
    st.warning(
        "Subjects need to be loaded from the Data Preprocessing tab. Please load from there before continuing. Thank You."
    )
else:
    selected_file = st.selectbox(
        "Select subject data to get EDA frames",
        st.session_state.uploaded_files_dict.keys(),
    )

    if selected_file:
        dataframe = st.session_state.uploaded_files_dict[selected_file]
        get_eda_using_profiling(dataframe=dataframe)
