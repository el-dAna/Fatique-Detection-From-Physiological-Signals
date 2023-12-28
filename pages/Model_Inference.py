import streamlit as st
# import boto3
from utils.rnn_predict import predict_from_streamlit_data


from mylib.appfunctions import (
    necessary_variables_app,
    #get_s3_bucket_files,
    get_s3_bucket_tagged_files,
    download_s3_file,
)


st.set_page_config(page_title="Run Inference", page_icon="😎")

st.markdown("# Run Inference")
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
    "selected_subjects_during_datapreprocessing": " ",
    "selected_inference_subjects": " ",
    "selected_model": " ",
    "sampling_window": 60,
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
}


@st.cache_data
def initialise_session_states():
    for key, value in session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_states()

# Create three columns to arrange the text inputs horizontally
col1, col2, col3 = st.columns(3)

# Create text input widgets in each column
st.session_state.sample_window = col1.number_input(
    "Preferred sampling window of data used to train models saved on s3:",
    min_value=0,
    max_value=500,
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
st.session_state.selected_inference_subjects = st.multiselect(
    "Select subject to run inference", st.session_state.uploaded_subject_names
)

# # bucket_name = 'your_bucket_name'
# tags = [{'Key': 'window', 'Value': str(st.session_state.sample_window)}, {'Key': 'overlap', 'Value': str(st.session_state.degree_of_overlap)}]


if (
    st.session_state.sample_window
    and st.session_state.degree_of_overlap
    and st.session_state.selected_inference_subjects
):
    # st.write("selected_inference", st.session_state.selected_inference_subjects)
    total_selected = len(st.session_state.selected_inference_subjects)
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
        sampling_window=st.session_state.sample_window,
        degree_of_overlap=st.session_state.degree_of_overlap,
    )  # get_s3_bucket_files(bucket_name="physiologicalsignalsbucket")
    st.selected_model = st.selectbox(
        "Select a(your) trained and saved model from s3 for inference",
        options=selected_models_on_s3,
    )

    if st.selected_model != " ":
        download_s3_file(s3_file_path=st.selected_model)
        model_local_path = "./temp/models/downloaded_model.h5"

        Confusion_matrix = predict_from_streamlit_data(
            inference_model=model_local_path,
            streamlit_all_data_dict=st.session_state.ALL_DATA_DICT,
            WINDOW=st.session_state.sampling_window,
            OVERLAP=st.session_state.degree_of_overlap,
        )
        st.write(Confusion_matrix)

        if st.button("Train model", type="primary"):
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
                help="Degree of intersection between samples, 0 means no intersection and 1 means full intersection(meaning sampling the same item). So max should be 0.9, thus 90 percent intersection",
            )
            st.session_state.sampling_window = st.number_input(
                "Sampling window:", min_value=100, max_value=500, value="min", step=10
            )
