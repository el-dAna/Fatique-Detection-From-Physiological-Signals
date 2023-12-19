import streamlit as st

from utils.rnn_predict import predict_from_streamlit_data
from utils.clearmlsdk_functions import clearml_model_list

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
    This is an mlops portforlio project. The project uses data collected from participants and a machine learning model trained to detecth fatigue.
    """
)

uploaded_files_dict = upload_files(from_s3=True)

if uploaded_files_dict:
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

    st.session_state.selected_inference_subjects = st.multiselect(
        "Select subject to run inference", st.session_state.uploaded_subject_names
    )

    if st.session_state.selected_inference_subjects:
        st.write("selected_inference", st.session_state.selected_inference_subjects)
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

        SPO2HR_resized, AccTempEDA_resized = resize_data_to_uniform_lengths_app(
            total_subject_num=total_selected,
            categories=categories,
            attributes_dict=attributes_dict,
            SPO2HR_target_size=SPO2HR_target_size,
            SPO2HR=SPO2HR,
            AccTempEDA_target_size=AccTempEDA_target_size,
            AccTempEDA=AccTempEDA,
        )

        write_expandable_text_app(
            title="SPO2HR_resized",
            detailed_description="Details",
            variable=SPO2HR_resized,
        )

        AccTempEDA_DownSampled = sanity_check_2_and_DownSamplingAccTempEDA_app(
            total_selected,
            categories,
            attributes_dict,
            SPO2HR_target_size,
            SPO2HR_resized,
            AccTempEDA_target_size,
            AccTempEDA_resized,
            relax_indices,
            phy_emo_cog_indices,
        )

        write_expandable_text_app(
            title="AccTempEDA_DownSampled",
            detailed_description="More details",
            variable=AccTempEDA_DownSampled,
        )

        ALL_DATA_DICT = get_data_dict_app(
            total_selected,
            categories,
            attributes_dict,
            SPO2HR_resized,
            AccTempEDA_DownSampled,
        )

        LABELS_TO_NUMBERS_DICT = {j: i for i, j in enumerate(categories)}
        NUMBERS_TO_LABELS_DICT = {i: j for i, j in enumerate(categories)}

        write_expandable_text_app(
            title="All_DATA_DICT",
            detailed_description="More details",
            variable=ALL_DATA_DICT,
        )

        st.write(
            "For every subject, there are 4 Relax sessions and just 1 session fot the other classes. This makes the ratio of Relax to any given class 4:1. The All_DATA_DICT stores the extracted values for the sessions. The keys of the dict do not represent the subject number. The keys are only indices of the samples generated. If only 1 subject, 7 samples are extracted(first 4 for Relax and the last 3 for the physical, emotional cognitive stress in that order)."
        )

        st.selected_model = st.selectbox(
            "Select a trained and saved model on clearml for inference",
            options=clearml_model_list,
        )
        if st.selected_model != " ":
            Confusion_matrix = predict_from_streamlit_data(
                inference_model=st.selected_model.get_local_copy(),
                streamlit_all_data_dict=ALL_DATA_DICT,
            )
            st.write(Confusion_matrix)


st.sidebar.markdown("# Data ❄️")


st.sidebar.markdown("# Model❄️")
