import streamlit as st

from utils.rnn_predict import predict_from_streamlit_data

from utils.rnn_train import (
    init_clearml_task,
    initialise_training_variables,
    initialise_train_model,
    train_model,
    save_trained_model_s3bucket_and_log_artifacts,
    get_trained_model_confusionM,
)

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


st.set_page_config(page_title="Run Inference", page_icon="😎")

st.markdown("# Plotting Demo")
st.sidebar.header("Variables to track")
st.write(
    """This page is for classifying the samples of subjects loaded from s3 bucket"""
)




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

