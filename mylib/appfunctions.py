import streamlit as st

# import os
import pandas as pd

# from tempfile import NamedTemporaryFile
# import seaborn as sns
import matplotlib.pyplot as plt

# import boto3
# from io import BytesIO


def upload_files():
    """
    Function to upload files using Streamlit file_uploader.

    Returns:
    - dict or []: dict of file names and uploaded file object from the streamlit file_uploader or [] if no files are uploaded.
    """

    file_names = []
    uploaded_files_dict = {}
    uploaded_files = st.file_uploader(
        "Upload files", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            file_names.append(file.name)
            uploaded_files_dict[file.name] = file

        if len(file_names) % 2 != 0:
            st.error("Please upload an even number of files.")
            return None
        return uploaded_files_dict

    return []


def read_files(uploaded_files_dict):
    """
    Function to read CSV files from and plot data based on the selected files.

    Args:
    - uploaded_filees_dict (dict): dict of file names as values and uploaded file object from calling st.file_uploader().

    Returns:
    - None: List of DataFrames or None if an error occurs.
    """

    # selected_file = st.selectbox("Select an option:", uploaded_files_dict.keys())

    selected_files = st.multiselect("Select subject data", uploaded_files_dict.keys())
    if selected_files:
        # Create columns for plots
        graph_cols = st.columns(int(len(selected_files)))
        pandas_cols = st.columns(int(len(selected_files)))

        for i, file in enumerate(selected_files):
            with pandas_cols[i]:
                # Create pandas dataframe
                selected_file_1 = uploaded_files_dict[file]
                dataframe = pd.read_csv(selected_file_1)
                st.write(dataframe)
                time_steps = range(len(dataframe["Second"]))
            with graph_cols[i]:
                if "EDA" in file:
                    # dataframe.plot(x='Second', y=['AccX', 'AccY', 'AccZ', 'Temp', 'EDA'])
                    plt.plot(
                        time_steps,
                        dataframe["AccX"],
                        time_steps,
                        dataframe["AccY"],
                        time_steps,
                        dataframe["AccZ"],
                        time_steps,
                        dataframe["Temp"],
                        time_steps,
                        dataframe["EDA"],
                    )
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    st.pyplot(plt)
                    plt.close()
                else:
                    # dataframe.plot(x='Second', y=['HeartRate', 'SpO2'])
                    plt.plot(
                        time_steps,
                        dataframe["HeartRate"],
                        time_steps,
                        dataframe["SpO2"],
                    )
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    st.pyplot(plt)
                    plt.close()