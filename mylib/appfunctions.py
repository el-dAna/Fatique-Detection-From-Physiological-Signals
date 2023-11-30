import streamlit as st
from dataclasses import dataclass

# import os
import pandas as pd

# from tempfile import NamedTemporaryFile
# import seaborn as sns
import matplotlib.pyplot as plt

# import boto3
# from io import BytesIO


@dataclass
class TEXT:
    dataset_description1 = "As shown in the graphs above the total recoding time for AccTempEDA and SpO2HR files is diiferent. The signals were sampled at different frequencies. One other challenge is that sessions for Relax, PhysicalStress, CognitiveStress, EmotionalStress are all contained in one file. So to have distinct classes each needs to be extracted."
    # https://physionet.org/content/noneeg/1.0.0/


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
                dataframe['Second_modified'] = time_steps
            with graph_cols[i]:
                if "EDA" in file:
                    dataframe.plot(x='Second_modified', y=['AccX', 'AccY', 'AccZ', 'Temp', 'EDA'])
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    st.pyplot(plt)
                    plt.close()
                else:
                    dataframe.plot(x='Second_modified', y=['HeartRate', 'SpO2'])
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    st.pyplot(plt)
                    plt.close()


def get_numerical_labels(dataframe):
    labels = dataframe['Labels']
    # (dataframe.Labels.values == 'PhysicalStress').argmin()
    # (dataframe.Labels.values == 'CognitiveStress').argmin()
    # (dataframe.Labels.values == 'EmotionalStress').argmin()
    labels = ['Relax', 'PhysicalStress', 'CognitiveStress', 'EmotionalStress']
    labels_dict = dict(zip(labels, range(0,4)))
    dataframe['labels'].map(labels_dict, na_action='ignore')
    return dataframe


def group_dataframe_by(dataframe, column_name="Label"):
    session_grp = dataframe.groupby(column_name)
    return session_grp
    