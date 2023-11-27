import streamlit as st
import os
import pandas as pd


def upload_files():
    """
    Function to upload files using Streamlit file_uploader.

    Returns:
    - list or None: List of uploaded files or None if no files are uploaded.
    """
    uploaded_files = st.file_uploader("Upload files", type="csv", accept_multiple_files=True)

    if uploaded_files:
        st.write("Uploaded files:")
        file_names = [file.name for file in uploaded_files]
        st.write("\n".join(file_names))
        file_paths = [os.path.join(tempfile.gettempdir(), file.name) for file in uploaded_files]

        # file_name = uploaded_file.name
        # file_path = os.path.join(tempfile.gettempdir(), file_name)
        # st.write(f"File path: {file_path}")

        if len(file_paths) % 2 != 0:
            st.error("Please upload an even number of files.")
            return None

        return file_paths

    return None



def read_files(file_paths):
    """
    Function to read CSV files from the given file paths.

    Args:
    - file_paths (list): List of file paths.

    Returns:
    - list or None: List of DataFrames or None if an error occurs.
    """

    selected_file = st.selectbox("Select an option:", file_paths)

    try:
        data = pd.read_csv(selected_file)
        st.write(data.describe())
    except Exception as e:
        st.error(f"Error reading file {selected_file}: {e}")
        return None