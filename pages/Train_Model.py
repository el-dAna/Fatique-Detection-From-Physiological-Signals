import streamlit as st

# import tensorflow as tf
# import datetime

from utils.rnn_train import (
    # init_clearml_task,
    # initialise_training_variables,
    # initialise_train_model,
    # train_model,
    # save_trained_model_s3bucket_and_log_artifacts,
    # get_trained_model_confusionM,
    train_new_model_from_streamlit_ui,
)

st.set_page_config(page_title="Run Inference", page_icon="😎")

st.markdown("# Plotting Demo")
st.sidebar.header("Variables to track")
st.write(
    """This page is for classifying the samples of subjects loaded from s3 bucket"""
)

st.write("All session states", st.session_state)

st.session_state.PERCENT_OF_TRAIN = st.slider(
    "Percentage of train samples:",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1,
    help="Percent of total samples for training. 0 is no sample for training and 1 means all samples for training. 0 training samples is illogical so min kept at 0.1 thus 10 percent.",
)
st.session_state.degree_of_overlap = st.number_input(
    "Degree of overlap between two consecutive samples:",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help="Degree of intersection between samples, 0 means no intersection and 1 means full intersection(meaning sample the same item). So max should be 0.9, thus 90 percent intersection",
)
st.session_state.sample_window = st.number_input(
    "Sampling window:", min_value=100, max_value=500, value="min", step=10
)
st.session_state.EPOCHS = st.number_input(
    "Number of epochs:", min_value=10, max_value=None, value="min", step=1
)

# Create three columns to arrange the text inputs horizontally
col1, col2, col3 = st.columns(3)

# Create text input widgets in each column
st.session_state.clearml_task_name = col1.text_input("Clearml task name:")
st.session_state.model_s3_name = col2.text_input("Name of model to save in s3:")
st.session_state.LOSS = st.selectbox(
    "Select tf loss function to use",
    options=["tf.keras.losses.Huber()", "tf.keras.losses.CategoricalCrossentropy()"],
)
st.session_state.learning_rate = col3.number_input(
    "Enter the learning rate:", min_value=0.0, max_value=1.0, value=0.0002, step=0.0001
)

if st.button("Train model", type="primary"):
    train_new_model_from_streamlit_ui(
        clearml_task_name=st.session_state.clearml_task_name,
        sample_window=st.session_state.sample_window,
        degree_of_overlap=st.session_state.degree_of_overlap,
        PERCENT_OF_TRAIN=st.session_state.PERCENT_OF_TRAIN,
        learning_rate=st.session_state.learning_rate,
        LOSS=st.session_state.LOSS,
        EPOCHS=st.session_state.EPOCHS,
        model_s3_name=st.session_state.model_s3_name,
    )
