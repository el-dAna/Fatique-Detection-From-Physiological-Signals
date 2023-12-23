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

st.markdown("# Understanding Data and Preprocessing.")
st.sidebar.header("Variables to track")
st.write(
    """This page is for understanding the properties of the recorded signals and preprocessing the samples of specific subjects loaded from s3 bucket."""
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



uploaded_files_dict = upload_files(from_s3=True)

if uploaded_files_dict:
    write_expandable_text_app(title='More info', detailed_description="""Total subject number is 20. Eavh subject has 2 files so a total of 2*20=40 files.
                              The Subject#SpO2HR.csv has SpO2 and HeartRate information. The Subject#AccTempEDA.csv contains
                              the Acceleration in (X, Y, Z) directions,Temperature and Electrodermal Activity(EDA) of the subject
                            """)

    st.session_state.files_upload = True
    st.session_state.uploaded_files_dict = uploaded_files_dict
    st.session_state.uploaded_files_dict_keys = (
        st.session_state.uploaded_files_dict.keys()
    )
    st.session_state.uploaded_subject_names = set(
        [key[:8] for key in st.session_state.uploaded_files_dict_keys]
    )


if st.session_state.files_upload:
    # view_data_button = st.button("View")
    read_files(uploaded_files_dict=st.session_state.uploaded_files_dict)
    st.write(TEXT.dataset_description1)
    st.session_state.uploaded_spo2_files = [
        file
        for file in st.session_state.uploaded_files_dict_keys
        if file.endswith("HR.csv")
    ]
    st.session_state.uploaded_tempEda_files = [
        file
        for file in st.session_state.uploaded_files_dict_keys
        if file.endswith("EDA.csv")
    ]
    SPO2HR, SPO2HR_attributes_dict = SortSPO2HR_app(
        uploaded_files_dict=st.session_state.uploaded_files_dict,
        uploaded_spo2_files=st.session_state.uploaded_spo2_files,
    )
    AccTempEDA, AccTempEDA_attributes_dict = SortAccTempEDA_app(
        uploaded_files_dict=st.session_state.uploaded_files_dict,
        uploaded_tempEda_files=st.session_state.uploaded_tempEda_files,
    )

    st.session_state.selected_subjects_during_datapreprocessing = st.multiselect(
        "Select subject to restructure classes in a training/inference format", st.session_state.uploaded_subject_names
    )

    if st.session_state.selected_subjects_during_datapreprocessing:
        st.write("selected_inference", st.session_state.selected_subjects_during_datapreprocessing)
        total_selected = len(st.session_state.selected_subjects_during_datapreprocessing)
        (
            st.session_state.SPO2HR_target_size,
            st.session_state.AccTempEDA_target_size,
            st.session_state.SPO2HR_attributes,
            st.session_state.AccTempEDA_attributes,
            st.session_state.categories,
            st.session_state.attributes_dict,
            st.session_state.relax_indices,
            st.session_state.phy_emo_cog_indices,
            st.session_state.all_attributes,
        ) = necessary_variables_app()

        st.session_state.SPO2HR_resized, st.session_state.AccTempEDA_resized = resize_data_to_uniform_lengths_app(
            total_subject_num=total_selected,
            categories=st.session_state.categories,
            attributes_dict=st.session_state.attributes_dict,
            SPO2HR_target_size=st.session_state.SPO2HR_target_size,
            SPO2HR=SPO2HR,
            AccTempEDA_target_size=st.session_state.AccTempEDA_target_size,
            AccTempEDA=AccTempEDA,
        )

        write_expandable_text_app(
            title="SPO2HR_resized",
            detailed_description="Details",
            variable=st.session_state.SPO2HR_resized,
        )

        st.session_state.AccTempEDA_DownSampled = sanity_check_2_and_DownSamplingAccTempEDA_app(
            total_selected,
            st.session_state.categories,
            st.session_state.attributes_dict,
            st.session_state.SPO2HR_target_size,
            st.session_state.SPO2HR_resized,
            st.session_state.AccTempEDA_target_size,
            st.session_state.AccTempEDA_resized,
            st.session_state.relax_indices,
            st.session_state.phy_emo_cog_indices,
        )

        write_expandable_text_app(
            title="AccTempEDA_DownSampled",
            detailed_description="More details",
            variable=st.session_state.AccTempEDA_DownSampled,
        )

        st.session_state.ALL_DATA_DICT = get_data_dict_app(
            total_selected,
            st.session_state.categories,
            st.session_state.attributes_dict,
            st.session_state.SPO2HR_resized,
            st.session_state.AccTempEDA_DownSampled,
        )

        st.session_state.LABELS_TO_NUMBERS_DICT = {j: i for i, j in enumerate(st.session_state.categories)}
        st.session_state.NUMBERS_TO_LABELS_DICT = {i: j for i, j in enumerate(st.session_state.categories)}

        write_expandable_text_app(
            title="All_DATA_DICT",
            detailed_description="More details",
            variable=st.session_state.ALL_DATA_DICT,
        )

        st.write(
            "For every subject, there are 4 Relax sessions and just 1 session fot the other classes. This makes the ratio of Relax to any given class 4:1. The All_DATA_DICT stores the extracted values for the sessions. The keys of the dict do not represent the subject number. The keys are only indices of the samples generated. If only 1 subject, 7 samples are extracted(first 4 for Relax and the last 3 for the physical, emotional cognitive stress in that order)."
        )

