import streamlit as st
import tensorflow as tf
# import boto3
# import datetime
from utils.rnn_predict import predict_from_streamlit_data


from mylib.appfunctions import (
    necessary_variables_app,
    # get_s3_bucket_files,
    get_s3_bucket_tagged_files,
    download_s3_file,
    write_expandable_text_app,
)

from utils.rnn_train import train_new_model_from_streamlit_ui, init_clearml_task

st.set_page_config(page_title="Run Inference", page_icon="😎")

st.markdown("# Run Inference")
st.sidebar.header("Variables to track")
st.write(
    """This page is for classifying the samples of subjects loaded from s3 bucket\n\n
    Each feature has shape(7,300).\n
    7 because (SpO2, HeartRate, AccX, AccY, AccZ, Temp, EDA)\n
    300 becuase a value every second for 5mins = 300 values.\n
    To resample from this (7,300) using a sample_window of 100 and overlap of 0.5 generates [(300-100)/(0.5*100)]+1 = 5 samples.\n
    For example: ALL_DATA_DICT[0].shape = (7,300). This is an original Relax sample. We can use this for training/inference, but to generate more samples we have to resample.\n
    So set sample_window = w, overlap = o, and then (300-w)/o*w + 1 samples are generated for this original Relax sample only.\n
    This is done for all original samples while keeping the indices to know the labels.
    """
)


# Create three columns to arrange the text inputs horizontally
col1, col2, col3 = st.columns(3)

# Create text input widgets in each column
st.session_state.sample_window = col1.number_input(
    "Preferred sample window of data used to train models saved on s3:",
    min_value=50,
    max_value=300,
    value=100,
    step=1,
)
st.session_state.degree_of_overlap = col2.number_input(
    "Preferred overlap between sampled windows used to train models:",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.1,
)

st.session_state.sample_per_sample = int(((300-st.session_state.sample_window)/(st.session_state.degree_of_overlap*st.session_state.sample_window))+1)        

if st.session_state.uploaded_files_dict == 0:
    st.warning(
        "Subjects need to be loaded from the Data Preprocessing tab. Please load from there before continuing. Thank You."
    )


# # bucket_name = 'your_bucket_name'
# tags = [{'Key': 'window', 'Value': str(st.session_state.sample_window)}, {'Key': 'overlap', 'Value': str(st.session_state.degree_of_overlap)}]


if (
    st.session_state.sample_window
    and st.session_state.degree_of_overlap
    and st.session_state.ALL_DATA_DICT != 0
):
    write_expandable_text_app(
            title="Check how many samples are generated!",
            detailed_description=f"""\n
            INTEGER VALUES TAKEN.\n
            A sample window of {int(st.session_state.sample_window)} with an overlap of {int(st.session_state.degree_of_overlap)} over a width if 300 generates {int(st.session_state.sample_per_sample)} samples per each original (7,300) sample.\n
            This generates a total of {int(st.session_state.sample_per_sample*len(st.session_state.ALL_DATA_DICT.keys()))} samples.\n
            From {int(len(st.session_state.ALL_DATA_DICT.keys())/7)} subject(s) data uploaded, each subject had 7  (Relax1,Relax2,Relax3,Relax4,PhysicalStress,CognitiveStress,EmotionalStress) original samples.\n
            {int(st.session_state.sample_per_sample)} samples were generated from each of the 7. This totals {int(st.session_state.sample_per_sample*len(st.session_state.ALL_DATA_DICT.keys()))} samples. Got it now?
            """)
    (
        SPO2HR_target_size,
        AccTempEDA_target_size,
        SPO2HR_attributes,
        AccTempEDA_attributes,
        categories,
        attributes_dict,
        relax_indices,
        phy_emo_cog_indices,
        all_attributes,
    ) = necessary_variables_app()

    selected_models_on_s3 = get_s3_bucket_tagged_files(
        sample_window=st.session_state.sample_window,
        degree_of_overlap=st.session_state.degree_of_overlap,
    )  # get_s3_bucket_files(bucket_name="physiologicalsignalsbucket")
    if selected_models_on_s3 != None:
        st.selected_model = st.selectbox(
            "Select a(your) trained and saved model from s3 for inference",
            options=[" "] + selected_models_on_s3,
        )

        if st.selected_model != " ":
            
            model_local_path = download_s3_file(s3_file_path=st.selected_model)
            #st.write('Model Inference', st.session_state.sample_window )
            Confusion_matrix = predict_from_streamlit_data(
                inference_model=model_local_path,
                streamlit_all_data_dict=st.session_state.ALL_DATA_DICT,
                WINDOW=st.session_state.sample_window,
                OVERLAP=st.session_state.degree_of_overlap,
            )
            st.write(Confusion_matrix)
            total_samples_for_each_class = Confusion_matrix.sum(axis=1)
            write_expandable_text_app(
            title="RESULTS INTERPRETATION",
            detailed_description=f"""\n
            Relax: {Confusion_matrix[0,0]} were accurate, Accuracy: {Confusion_matrix[0,0]/total_samples_for_each_class[0]}\n
            PhysicalStress: {Confusion_matrix[1,1]} were accurate, Accuracy: {Confusion_matrix[1,1]/total_samples_for_each_class[1]}\n
            CognitiveStress: {Confusion_matrix[2,2]} were accurate, Accuracy: {Confusion_matrix[2,2]/total_samples_for_each_class[2]}\n
            EmotionalStress: {Confusion_matrix[3,3]} were accurate, Accuracy: {Confusion_matrix[3,3]/total_samples_for_each_class[3]}\n
            """)
    else:
        if st.button("Train model with spececifications above.", type="primary"):
            st.session_state.PERCENT_OF_TRAIN = st.slider(
                "Percentage of train samples:",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Percent of total samples for training. 0 is no sample for training and 1 means all samples for training. 0 training samples is illogical so min kept at 0.1 thus 10 percent.",
            )
            st.session_state.clearml_task_name = col1.text_input("Clearml task name:")
            st.session_state.model_s3_name = col2.text_input(
                "Name of model to save in s3:"
            )
            st.session_state.LOSS = st.selectbox(
                "Select tf loss function to use",
                options=[st.session_state.LOSSES.keys()],
            )
            st.session_state.learning_rate = col3.number_input(
                "Enter the learning rate:",
                min_value=0.0,
                max_value=1.0,
                value=0.0002,
                step=0.0001,
            )
            st.session_state.EPOCHS = st.number_input(
                "Number of epochs:", min_value=10, max_value=None, value="min", step=1
            )

            try:
                st.session_state.train_task.close()
                st.session_state.train_task = init_clearml_task(task_name=st.session_state.clearml_task_name)
            except Exception:
                pass

            if st.button("Train model", type="primary"):
                st.session_state.train_task = train_new_model_from_streamlit_ui(
                    train_task=st.session_state.train_task,
                    clearml_task_name=st.session_state.clearml_task_name,
                    sample_window=st.session_state.sample_window,
                    degree_of_overlap=st.session_state.degree_of_overlap,
                    PERCENT_OF_TRAIN=st.session_state.PERCENT_OF_TRAIN,
                    learning_rate=st.session_state.learning_rate,
                    LOSS=st.session_state.LOSSES[st.session_state.LOSS],
                    EPOCHS=st.session_state.EPOCHS,
                    model_s3_name=st.session_state.model_s3_name,
                )
            # st.session_state.degree_of_overlap = st.number_input(
            #     "Degree of overlap between two consecutive samples:",
            #     min_value=0.0,
            #     max_value=0.9,
            #     value=0.5,
            #     step=0.1,
            #     help="Degree of intersection between samples, 0 means no intersection and 1 means full intersection(meaning sample the same item). So max should be 0.9, thus 90 percent intersection",
            # )
            # st.session_state.sample_window = st.number_input(
            #     "Sampling window:", min_value=100, max_value=500, value="min", step=10
            # )
