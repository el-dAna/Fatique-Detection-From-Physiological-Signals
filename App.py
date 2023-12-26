import streamlit as st

from utils.rnn_predict import predict_from_streamlit_data

from mylib.appfunctions import (
    upload_files,
    read_files,
    TEXT,
    SortSPO2HR_app,
    SortAccTempEDA_app,
    necessary_variables_app,
    resize_data_to_uniform_lengths_app,
    write_expandable_text_app,
    sanity_check_2_and_DownSamplingAccTempEDA_app,
    get_data_dict_app,
    get_s3_bucket_files,
    download_s3_file,
)


# import numpy as np
session_states = {
    "files_upload": False,
    "uploaded_files_dict": 0,
    "uploaded_files_dict_keys": 0,
    "uploaded_spo2_files": 0,
    "uploaded_tempEda_files": 0,
    "uploaded_subject_names": 0,
    "selected_inference_subjects": " ",
    "selected_model": " ",
    "sampling_window": 60,
    "degree_of_overlap": 0.5,
    "PERCENT_OF_TRAIN": 0.8,
}


@st.cache_data
def initialise_session_states():
    for key, value in session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_states()
# st.write("All session states", st.session_state)

st.markdown("# Fatigue Detection from Physiological Signals🎈")
st.sidebar.markdown("# Home Page 🎈")


st.markdown(
    """
    This is an mlops portforlio project. The project uses physiological signals collected from 20 participants and a machine learning model trained to detect fatigue.
    This project shows more of mlops operations than model accuracy. A model can be trained given the timea resources.
    """
)

st.markdown("## Dataset Description")

st.markdown("""This dataset consists of physiological signals taken from 20 participants as they performed various tasks. Therea are 4 classes. \n
            (Relax, Physical Stress, Cognitive Stress and Emotional Stress)\n
            Check the table below for a summary of protocol.
            """)

data = [
    ["Time", "State", "Description", "Files"],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
    ["5 mins", "Physical Stress", "Stand, Walk, Jog on Treadmill", "AccTempEDA.csv and SpO2HR.csv"],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
    ["40 secs", "Mini CognitiveStress", "Instructions for next session read", "AccTempEDA.csv and SpO2HR.csv"],
    ["5 mins", "Cognitive Stress", " Count backwards by sevens from 2485", "AccTempEDA.csv and SpO2HR.csv"],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
    ["5 mins", "Emotional Stress", "Watching a clip from zombie apocalypse", "AccTempEDA.csv and SpO2HR.csv"],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
]

st.table(data)

st.markdown(
    """
    ###### 20 subjects were used.\n
    ###### Each subject had two (AccTempEDA.csv and SpO2HR.csv) files.
    ###### AccTempEDA.csv contains Acceleration(X,Y,Z), Temperature and Electrodermal Activity(EDA). Sampled at 8hz
    ###### SpO2HR.csv contains Oxygen saturation(SpO2) and Heart Rate data. Sampled at 8Hz.
    """
)

st.markdown("Find more information and access the dataset from here (https://physionet.org/content/noneeg/1.0.0/#files-panel)")

st.sidebar.markdown("# Data Preprocessing ❄️")

st.sidebar.markdown("# Model Taining❄️")

st.sidebar.markdown("# Inference❄️")
