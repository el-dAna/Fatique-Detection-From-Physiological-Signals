import streamlit as st
from dataclasses import dataclass
from utils.rnn_predict import predict_from_streamlit_data
import datetime
import tensorflow as tf

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
    "loss_function": 0,
    "learning_rate": 0.0002,
    "EPOCHS": 10,
}

@dataclass
class TRAIN_MODEL:
    clearml_project_name = "portfolioproject"
    
    # Get the current date and time
    current_datetime = str(datetime.datetime.now())
    clearml_task_name= f"task-{current_datetime}"
    model_local_path="./data/models/model.h5"
    bucket_name='physiologicalsignalsbucket',
    model_s3_name=f"Model-{current_datetime}"

    number_of_classes=4




@st.cache_data
def initialise_session_states():
    for key, value in session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_states()





st.session_state.PERCENT_OF_TRAIN = st.slider('Percentage of train samples:', min_value=0.1, max_value=1.0, value=0.8, step=0.1, help="Percent of total samples for training. 0 is no sample for training and 1 means all samples for training. 0 training samples is illogical so min kept at 0.1 thus 10 percent.")
st.session_state.degree_of_overlap = st.number_input('Degree of overlap between two consecutive samples:', min_value=0.0, max_value=0.9, value=0.5, step=0.1, help="Degree of intersection between samples, 0 means no intersection and 1 means full intersection(meaning sampling the same item). So max should be 0.9, thus 90 percent intersection" )
st.session_state.sampling_window = st.number_input('Sampling window:', min_value=100, max_value=500, value="min", step=10)
st.session_state.EPOCHS = st.number_input('Number of epochs:', min_value=10, max_value=None, value="min", step=1)

# Create three columns to arrange the text inputs horizontally
col1, col2, col3 = st.columns(3)

# Create text input widgets in each column
st.session_state.clearml_task_name = col1.text_input("Clearml task name:")
st.session_state.model_s3_name = col2.text_input("Name of model to save in s3:")
st.session_state.loss_function = st.selectbox(
    "Select tf loss function to use",
    options=['tf.keras.losses.Huber()'],
)
st.session_state.learning_rate = col3.number_input("Enter the learning rate:", min_value=0.0, max_value=1.0, value=0.0002, step=0.0001)

if st.button("Train model", type="primary"):

    train_task = init_clearml_task(project_name=TRAIN_MODEL.clearml_project_name, task_name=st.session_state.clearml_task_name)

    (
        TRAIN_FEATURES,
        TRAIN_LABELS,
        TOTAL_TRAIN_DATA,
        PREDICT_FEATURES,
        PREDICT_LABELS,
        TOTAL_VAL_DATA,
        INPUT_FEATURE_SHAPE,
        TRAIN_BATCH_SIZE,
        TRAIN_STEPS,
    ) = initialise_training_variables(
        sample_window=st.session_state.sampling_window,
        degree_of_overlap=st.session_state.degree_of_overlap,
        PERCENT_OF_TRAIN=st.session_state.PERCENT_OF_TRAIN,
    )

    model_to_train = initialise_train_model(
        MODEL_INPUT_SHAPE=INPUT_FEATURE_SHAPE,
        dp1=0.3,
        dp2=0.3,
        dp3=0.0,
        dp4=0.0,
        learning_rate=st.session_state.learning_rate,
        LOSS= tf.keras.losses.Huber(), #st.session_state.loss_function,
        NUMBER_CLASSES=TRAIN_MODEL.number_of_classes
    )

    trained_model, history = train_model(
        model_to_train,
        TRAIN_FEATURES,
        TRAIN_LABELS,
        TRAIN_STEPS,
        PREDICT_FEATURES,
        PREDICT_LABELS,
        EPOCHS=st.session_state.EPOCHS,
    )

    save_trained_model_s3bucket_and_log_artifacts(
        trained_model,
        history,
        model_local_path= TRAIN_MODEL.model_local_path,
        bucket_name=TRAIN_MODEL.bucket_name,
        model_s3_name=TRAIN_MODEL.model_s3_name,
    )

    get_trained_model_confusionM(
        trained_model,
        TRAIN_FEATURES,
        TRAIN_LABELS,
        PREDICT_FEATURES,
        PREDICT_LABELS,
    )

    train_task.close()

    st.write("Done training and uploading")