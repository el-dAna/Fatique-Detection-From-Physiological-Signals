import streamlit as st
import datetime
import tensorflow as tf

session_states = {
    "files_upload": False,
    "uploaded_files_dict": 0,
    "uploaded_files_dict_keys": 0,
    "uploaded_spo2_files": 0,
    "uploaded_tempEda_files": 0,
    "uploaded_subject_names": 0,
    "selected_subjects_during_datapreprocessing": " ",
    "selected_inference_subjects": " ",
    "selected_model": " ",
    "sample_window": 60,
    "degree_of_overlap": 0.5,
    "PERCENT_OF_TRAIN": 0.8,
    "SPO2HR_target_size": 0,
    "AccTempEDA_target_size": 0,
    "SPO2HR_attributes": 0,
    "AccTempEDA_attributes": 0,
    "categories": 0,
    "attributes_dict": 0,
    "relax_indices": 0,
    "phy_emo_cog_indices": 0,
    "all_attributes": 0,
    "SPO2HR_resized": 0,
    "AccTempEDA_resized": 0,
    "AccTempEDA_DownSampled": 0,
    "ALL_DATA_DICT": 0,
    "LABELS_TO_NUMBERS_DICT": 0,
    "NUMBERS_TO_LABELS_DICT": 0,
    "learning_rate": 0.0002,
    "EPOCHS": 10,
    "LOSS": 0,
    "LOSSES": {
        "tf.keras.losses.Huber()": tf.keras.losses.Huber(),
        "tf.keras.losses.categorical_crossentropy()": tf.keras.losses.CategoricalCrossentropy(),
    },
    "sample_per_sample": 0,
    "train_task": 0,
}


@st.cache_data
def initialise_session_states():
    """
    Initializes session states in a Streamlit app.
    Note:
        Assumes the existence of a dictionary 'session_states'.
    """
    for key, value in session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_states()
#st.write("All session states", st.session_state)


st.markdown("# Fatigue Detection from Physiological Signals🎈")
st.sidebar.markdown("# Home Page 🎈")


st.markdown(
    """
    This is an mlops portforlio project. The project uses physiological signals collected from 20 participants and a machine learning model trained to detect fatigue.
    This project shows more of mlops operations than model accuracy. A model can be trained given the timea resources.
    """
)

st.markdown("### The structure if the orchestration is shown by the figma embed below")
figma_project_structure = """
        <iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="800" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FqAkiRCvXSZOOgIAfwgJiNY%2FTuseb%3Ftype%3Ddesign%26node-id%3D13%253A2%26mode%3Ddesign%26t%3Dh9lQy5zVoYmXZDEA-1" allowfullscreen></iframe>
    """
st.markdown(figma_project_structure, unsafe_allow_html=True)

st.markdown("## Dataset Description")

st.markdown(
    """This dataset consists of physiological signals taken from 20 participants as they performed various tasks. Therea are 4 classes. \n
            (Relax, Physical Stress, Cognitive Stress and Emotional Stress)\n
            Check the table below for a summary of protocol.
            """
)

data = [
    ["Time", "State", "Description", "Files"],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
    [
        "5 mins",
        "Physical Stress",
        "Stand, Walk, Jog on Treadmill",
        "AccTempEDA.csv and SpO2HR.csv",
    ],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
    [
        "40 secs",
        "Mini CognitiveStress",
        "Instructions for next session read",
        "AccTempEDA.csv and SpO2HR.csv",
    ],
    [
        "5 mins",
        "Cognitive Stress",
        " Count backwards by sevens from 2485",
        "AccTempEDA.csv and SpO2HR.csv",
    ],
    ["5 mins", "Relax", "Relaxing activities", "AccTempEDA.csv and SpO2HR.csv"],
    [
        "5 mins",
        "Emotional Stress",
        "Watching a clip from zombie apocalypse",
        "AccTempEDA.csv and SpO2HR.csv",
    ],
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


st.sidebar.markdown("# Data Preprocessing ❄️")

st.sidebar.markdown("# Model Taining❄️")

st.sidebar.markdown("# Inference❄️")
