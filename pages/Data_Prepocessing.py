import streamlit as st
from streamlit_extras.switch_page_button import switch_page

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


st.set_page_config(page_title="Run Inference", page_icon="😎")

st.markdown("# Understanding Data and Preprocessing.")
st.sidebar.header("Variables to track")
st.write(
    """This page is for understanding the properties of the recorded signals and preprocessing the samples of specific subjects loaded from s3 bucket."""
)


st.markdown("### Step 1. Loading")
uploaded_files_dict = upload_files(from_s3=True)
if uploaded_files_dict:
    write_expandable_text_app(
        title="More info on data",
        detailed_description="""Total subject number is 20. Eavh subject has 2 files so a total of 2*20=40 files.
                              The Subject#SpO2HR.csv has SpO2 and HeartRate information. The Subject#AccTempEDA.csv contains
                              the Acceleration in (X, Y, Z) directions,Temperature and Electrodermal Activity(EDA) of the subject
                            """,
    )

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

        st.markdown("### Step 2. Visualise Data")
        read_files(uploaded_files_dict=st.session_state.uploaded_files_dict)

        if st.button("More EDA on files👀", type="primary"):
            switch_page("More EDA")

        write_expandable_text_app(
            title="More info on graphs", detailed_description=TEXT.dataset_description1
        )

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

        st.markdown("### Step 3. Restructuring data")
        total_selected = len(st.session_state.uploaded_subject_names)
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

        (
            st.session_state.SPO2HR_resized,
            st.session_state.AccTempEDA_resized,
        ) = resize_data_to_uniform_lengths_app(
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
            detailed_description="""The input to a model must be uniform. Eventhough 5mins were recoded, there can be some dew seconds over/under recoding.
                This step ensures this uniformity for the SpO2HR.csv files only. For each subject there are 4 Relax sessions and just one (Physical/Cognitive/Emotional) stress.\n
                By Observation len(Relax) equals 3 times len(other_sessions)
                """,
            variable=st.session_state.SPO2HR_resized,
        )

        st.session_state.AccTempEDA_DownSampled = (
            sanity_check_2_and_DownSamplingAccTempEDA_app(
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
        )

        write_expandable_text_app(
            title="AccTempEDA_DownSampled",
            detailed_description="This step ensures the unifromity of recoreded sessions in the 8Hz AccTempEDA.csv file. This step also downsamples the 8Hz signals to 1HZ to conform with the structure of the SpO2HR.csv data. Again the relax sessions here 4 for every subject.",
            variable=st.session_state.AccTempEDA_DownSampled,
        )

        st.session_state.ALL_DATA_DICT = get_data_dict_app(
            total_selected,
            st.session_state.categories,
            st.session_state.attributes_dict,
            st.session_state.SPO2HR_resized,
            st.session_state.AccTempEDA_DownSampled,
        )

        st.session_state.LABELS_TO_NUMBERS_DICT = {
            j: i for i, j in enumerate(st.session_state.categories)
        }
        st.session_state.NUMBERS_TO_LABELS_DICT = {
            i: j for i, j in enumerate(st.session_state.categories)
        }

        st.markdown("### Step 4. Regrouping sessions.")
        write_expandable_text_app(
            title="All_DATA_DICT",
            detailed_description="All relax sessions extracted and put at knwon indices, same is done for Physical, Cognitive and Emotional Stresses.",
            variable=st.session_state.ALL_DATA_DICT,
        )

        st.write(
            "For every subject, there are 4 Relax sessions and just 1 session fot the other classes. This makes the ratio of Relax to any given class 4:1. The All_DATA_DICT stores the extracted values for the sessions. The keys of the dict do not represent the subject number. The keys are only indices of the samples generated. If only 1 subject, 7 samples are extracted(first 4 for Relax and the last 3 for the physical, emotional cognitive stress in that order)."
        )

        write_expandable_text_app(
            title="IMPORTANT NOTE!!",
            detailed_description="As you might have noticed there are 4times more samples for relax than for other sessions. This can cause CLASS IMBALANCE. Only the first session Relax cannot be used since model would not be able to classify other Relax sessions, for example, Relax 2 immediately after a physical Stress. SO WHAT DO WE DO? Check out the Datagenerator in CODE :)",
        )
