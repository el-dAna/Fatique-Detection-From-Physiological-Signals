import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from dataprep.eda import create_report

st.set_page_config(page_title="EXPLORATORY DATA ANALYSIS👀🔍✅ ", page_icon="🔍")
st.markdown("# Run Inference")
st.sidebar.header("Variables to track")


def get_eda_using_prep(selected_file):
    dataframe = st.session_state.uploaded_files_dict[selected_file]
    st.write(create_report(dataframe))
    st.write(dataframe)

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
        get_eda_using_prep(selected_file=selected_file)
