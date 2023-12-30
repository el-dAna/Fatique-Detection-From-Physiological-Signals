from dataclasses import dataclass
import streamlit as st

# import os
import pandas as pd
import numpy as np
import copy
import boto3


# from tempfile import NamedTemporaryFile
# import seaborn as sns
import matplotlib.pyplot as plt

# import boto3
# from io import BytesIO


# """
# Sessions outline

# Relax = 5*60 secs
# PhysicalStress = 6*60
# Relax = 5*60
# MiniCognitiveStress = 40
# CognitiveStress = 5*60
# Relax = 5*60
# EmotionalStress = 5*60
# Relax = 5*60
# """


@dataclass
class TEXT:
    dataset_description1 = "As shown in the graphs above the total recoding time for AccTempEDA and SpO2HR files is diiferent. The signals were sampled at different frequencies. One other challenge is that sessions for Relax, PhysicalStress, CognitiveStress, EmotionalStress are all contained in one file. So to have distinct classes each needs to be extracted."


@dataclass
class DATA_VARIABLES:
    Relax: int = 5
    PhysicalStress: int = 6
    MiniCognitiveStress: int = 40
    CognitiveStress: int = 5
    EmotionalStress: int = 5
    Seconds: int = 60

    @property
    def Total_time_minutes(self) -> float:
        return (
            self.Relax * 4
            + self.PhysicalStress
            + self.CognitiveStress
            + self.EmotionalStress
        ) + (self.MiniCognitiveStress / 60)

    @property
    def Total_time_seconds(self) -> float:
        return self.Total_time_minutes * 60

    @property
    def freq_eda_files(self) -> float:
        return 18230 / self.Total_time_seconds


def write_expandable_text_app(
    title, detailed_description, img_path=False, variable=False
):
    """
    Displays a collapsed image/variable (dict, list, dataframe).

    Args:
        title (str): Text to display on the collapsed bar.
        detailed_description (str): Detailed description to display when uncollapsed.
        img_path (str): Path of the image to show when uncollapsed.
        variable: Variable to display (if any).
    """
    with st.expander(title):
        st.write(detailed_description)
        if img_path:
            st.image(img_path)
        if variable:
            st.write(variable)


# def display_collapsed_dict_app(dictionary):
#     with st.expander("Dictionary"):
#         st.write(dictionary)


def upload_files(from_s3=False):
    """
    Function to upload files using Streamlit file_uploader.

    Args:
        from_s3 (bool): If True, uploads files from an S3 bucket.

    Returns:
        dict or []: A dict of file names and uploaded file objects from the Streamlit file_uploader or [] if no files are uploaded.
    """

    file_names = []
    uploaded_files_dict = {}

    # Check if uploading from S3
    if from_s3:
        for i in range(20):
            file_names.append("Subject" + str(i + 1))
        uploaded_files = st.multiselect(
            "Select subjects whose data you want to load", file_names
        )

    else:
        # Use file_uploader for local file uploads
        uploaded_files = st.file_uploader(
            "Upload files", type="csv", accept_multiple_files=True
        )

    if uploaded_files:
        for file in uploaded_files:
            if from_s3:
                # If uploading from S3, construct file paths and read CSV files
                bucket_name = "physiologicalsignals"
                folder_name = "HealthySubjectsBiosignalsDataSet"
                file_path1 = f"{bucket_name}/{folder_name}/{file}/{file}SpO2HR.csv"
                file_path2 = f"{bucket_name}/{folder_name}/{file}/{file}AccTempEDA.csv"

                data1 = pd.read_csv(f"s3://{file_path1}")
                data2 = pd.read_csv(f"s3://{file_path2}")

                uploaded_files_dict[f"{file}SpO2HR.csv"] = data1
                uploaded_files_dict[f"{file}AccTempEDA.csv"] = data2

            else:
                # If uploading locally, read CSV files
                file_names.append(file.name)
                uploaded_files_dict[file.name] = pd.read_csv(file)

        if not from_s3 and (len(file_names) % 2) != 0:
            st.error("Please upload an even number of files.")
            return []

    return uploaded_files_dict


def read_files(uploaded_files_dict):
    """
    Function to read and display data from uploaded files using Streamlit.

    Args:
        uploaded_files_dict (dict): A dictionary containing file names and corresponding data.

    Returns:
        None
    """

    # Allow the user to select multiple files
    selected_files = st.multiselect(
        "Select subject data. Max of 2 selections advised for a convenient display.",
        uploaded_files_dict.keys(),
    )

    if selected_files:
        # Create columns for plots and dataframes
        graph_cols = st.columns(int(len(selected_files)))
        pandas_cols = st.columns(int(len(selected_files)))

        for i, file in enumerate(selected_files):
            with pandas_cols[i]:
                # Create pandas dataframe
                dataframe = uploaded_files_dict[file]
                st.write(dataframe)
                time_steps = range(len(dataframe["Second"]))
                dataframe["Second_modified"] = time_steps

            with graph_cols[i]:
                # Create plots based on file type (EDA or others)
                if "EDA" in file:
                    dataframe.plot(
                        x="Second_modified", y=["AccX", "AccY", "AccZ", "Temp", "EDA"]
                    )
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    plot_vertical_lines(plot=plt, freq=8)
                    st.pyplot(plt)
                    plt.close()
                else:
                    dataframe.plot(x="Second_modified", y=["HeartRate", "SpO2"])
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    plot_vertical_lines(plot=plt)
                    st.pyplot(plt)
                    plt.close()


def plot_vertical_lines(plot, freq=1, seconds=DATA_VARIABLES.Seconds):
    """
    Function to plot vertical lines on a given plot.

    Args:
        plot: Matplotlib plot object.
        freq (int): Frequency factor.
        seconds (int): Duration in seconds for each activity, fetched from DATA_VARIABLES.

    Returns:
        None
    """

    plot.axvline(
        x=(5 * seconds) * freq, color="b", label="axvline - full height"
    )  # relax
    plot.axvline(
        x=(5 * seconds + 6 * seconds) * freq, color="r", label="axvline - full height"
    )  # physical stress
    plot.axvline(
        x=(5 * seconds + 6 * seconds + 5 * seconds) * freq,
        color="b",
        label="axvline - full height",
    )  # relax
    plot.axvline(
        x=(5 * seconds + 6 * seconds + 5 * seconds + 40) * freq,
        color="g",
        label="axvline - full height",
    )  # minicognitive
    plot.axvline(
        x=(5 * seconds + 6 * seconds + 5 * seconds + 40 + 5 * seconds) * freq,
        color="g",
        label="axvline - full height",
    )  # cognitive
    plot.axvline(
        x=(5 * seconds + 6 * seconds + 5 * seconds + 40 + 5 * seconds + 5 * seconds)
        * freq,
        color="b",
        label="axvline - full height",
    )  # relax
    plot.axvline(
        x=(
            5 * seconds
            + 6 * seconds
            + 5 * seconds
            + 40
            + 5 * seconds
            + 5 * seconds
            + 5 * seconds
        )
        * freq,
        color="y",
        label="axvline - full height",
    )  # emotional
    plot.axvline(
        x=(
            5 * seconds
            + 6 * seconds
            + 5 * seconds
            + 40
            + 5 * seconds
            + 5 * seconds
            + 5 * seconds
            + 5 * seconds
        )
        * freq,
        color="b",
        label="axvline - full height",
    )  # relax


# def get_numerical_labels(dataframe):
#     return dataframe


# def group_dataframe_by(dataframe, column_name="Label"):
#     session_grp = dataframe.groupby(column_name)
#     return session_grp


def SortSPO2HR_app(uploaded_files_dict, uploaded_spo2_files):
    """
    Process SpO2 and HeartRate data from different subjects based on their labels.

    Args:
        uploaded_files_dict (dict): Dictionary containing dataframes of uploaded files.
        uploaded_spo2_files (list): List of filenames corresponding to SpO2 and HeartRate data.

    Returns:
        tuple: A tuple containing two dictionaries:
            - SPO2HR: Organized data for each category (Relax, PhysicalStress, CognitiveStress, EmotionalStress)
                     with SpO2 and HeartRate.
            - SPO2HR_attributes_dict: Dictionary storing the length of each category from each subject.
    """

    SPO2HR = {
        "Relax": {"Spo2": [], "HeartRate": []},
        "PhysicalStress": {"Spo2": [], "HeartRate": []},
        "CognitiveStress": {"Spo2": [], "HeartRate": []},
        "EmotionalStress": {"Spo2": [], "HeartRate": []},
    }

    SPO2HR_attributes_dict = {
        "Relax": [],
        "CognitiveStress": [],
        "PhysicalStress": [],
        "EmotionalStress": [],
    }

    for file in uploaded_spo2_files:
        data2 = uploaded_files_dict[file]

        # Extracting the SpO2 and HeartRate columns of each subject
        spo2 = np.array(list(data2["SpO2"]))
        HeartRate = np.array(list(data2["HeartRate"]))
        labels = list(data2["Label"])  # the labels are strings!!
        labels_dict = {j: i for i, j in enumerate(set(labels))}

        # Initialize lists to store values for each category
        relax_spo2, relax_HeartRate = [], []
        cognitive_spo2, cognitive_HeartRate = [], []
        physical_spo2, physical_HeartRate = [], []
        emotional_spo2, emotional_HeartRate = [], []

        index = 0

        for j, i in enumerate(labels):
            # Append values to the appropriate category lists
            if i == "Relax":
                relax_spo2.append(spo2[j])
                relax_HeartRate.append(HeartRate[j])
            elif i == "CognitiveStress":
                cognitive_spo2.append(spo2[j])
                cognitive_HeartRate.append(HeartRate[j])
            elif i == "PhysicalStress":
                physical_spo2.append(spo2[j])
                physical_HeartRate.append(HeartRate[j])
            elif i == "EmotionalStress":
                emotional_spo2.append(spo2[j])
                emotional_HeartRate.append(HeartRate[j])
            else:
                print(f"Value not found. Index at {index}")
            index += 1

        # Check the equality of recorded values for SpO2 and HeartRate for each category
        assert len(relax_spo2) == len(relax_HeartRate)
        assert len(physical_spo2) == len(physical_HeartRate)
        assert len(emotional_spo2) == len(emotional_HeartRate)
        assert len(cognitive_spo2) == len(cognitive_HeartRate)
        assert len(relax_spo2) + len(physical_spo2) + len(emotional_spo2) + len(
            cognitive_spo2
        ) == len(labels)

        # Store the total recording time for each category from each subject
        SPO2HR_attributes_dict["Relax"].append(len(relax_spo2))
        SPO2HR_attributes_dict["PhysicalStress"].append(len(physical_spo2))
        SPO2HR_attributes_dict["CognitiveStress"].append(len(cognitive_spo2))
        SPO2HR_attributes_dict["EmotionalStress"].append(len(emotional_spo2))

        temp_dict = {
            "RelaxSpo2": relax_spo2,
            "RelaxHeartRate": relax_HeartRate,
            "PhysicalStressSpo2": physical_spo2,
            "PhysicalStressHeartRate": physical_HeartRate,
            "CognitiveStressSpo2": cognitive_spo2,
            "CognitiveStressHeartRate": cognitive_HeartRate,
            "EmotionalStressSpo2": emotional_spo2,
            "EmotionalStressHeartRate": emotional_HeartRate,
        }
        temp_list = ["Spo2", "HeartRate"]

        for (
            i
        ) in (
            labels_dict.keys()
        ):  # (Relax, PhysicalStress, CognitiveStress, EmotionalStress)
            for j in temp_list:  # ('Spo2', 'HeartRate')
                SPO2HR[i][j].append(temp_dict[i + j])

    return SPO2HR, SPO2HR_attributes_dict


def SortAccTempEDA_app(uploaded_files_dict, uploaded_tempEda_files):
    """
    Process accelerometer, temperature, and EDA data from different subjects based on their labels.

    Args:
        uploaded_files_dict (dict): Dictionary containing dataframes of uploaded files.
        uploaded_tempEda_files (list): List of filenames corresponding to accelerometer, temperature, and EDA data.

    Returns:
        tuple: A tuple containing two dictionaries:
            - AccTempEDA: Organized data for each category (Relax, PhysicalStress, CognitiveStress, EmotionalStress)
                          with accelerometer, temperature, and EDA data.
            - AccTempEDA_attributes_dict: Dictionary storing the length of each category from each subject.
    """

    AccTempEDA = {
        "Relax": {"AccZ": [], "AccY": [], "AccX": [], "Temp": [], "EDA": []},
        "PhysicalStress": {"AccZ": [], "AccY": [], "AccX": [], "Temp": [], "EDA": []},
        "CognitiveStress": {"AccZ": [], "AccY": [], "AccX": [], "Temp": [], "EDA": []},
        "EmotionalStress": {"AccZ": [], "AccY": [], "AccX": [], "Temp": [], "EDA": []},
    }

    AccTempEDA_attributes_dict = {
        "Relax": [],
        "PhysicalStress": [],
        "CognitiveStress": [],
        "EmotionalStress": [],
    }

    for file in uploaded_tempEda_files:
        # Extracting the (AccZ, AccY, AccX, Temp, EDA) columns of each subject file
        data1 = uploaded_files_dict[file]
        AccZ = list(data1["AccZ"])
        AccY = list(data1["AccY"])
        AccX = list(data1["AccX"])
        Temp = list(data1["Temp"])
        EDA = list(data1["EDA"])
        Label = list(data1["Label"])

        # Initialize lists to store extracts for specific categories
        Relax_AccY, Relax_AccX, Relax_AccZ, Relax_Temp, Relax_EDA = [], [], [], [], []
        physical_AccY, physical_AccX, physical_AccZ, physical_Temp, physical_EDA = (
            [],
            [],
            [],
            [],
            [],
        )
        (
            emotional_AccY,
            emotional_AccX,
            emotional_AccZ,
            emotional_Temp,
            emotional_EDA,
        ) = ([], [], [], [], [])
        (
            cognitive_AccY,
            cognitive_AccX,
            cognitive_AccZ,
            cognitive_Temp,
            cognitive_EDA,
        ) = ([], [], [], [], [])

        temp_dict1 = {
            "AccX": AccX,
            "AccY": AccY,
            "AccZ": AccZ,
            "Temp": Temp,
            "EDA": EDA,
        }
        temp_dict2 = {
            "RelaxAccY": Relax_AccY,
            "RelaxAccX": Relax_AccX,
            "RelaxAccZ": Relax_AccZ,
            "RelaxTemp": Relax_Temp,
            "RelaxEDA": Relax_EDA,
            "PhysicalStressAccY": physical_AccY,
            "PhysicalStressAccX": physical_AccX,
            "PhysicalStressAccZ": physical_AccZ,
            "PhysicalStressTemp": physical_Temp,
            "PhysicalStressEDA": physical_EDA,
            "EmotionalStressAccY": emotional_AccY,
            "EmotionalStressAccX": emotional_AccX,
            "EmotionalStressAccZ": emotional_AccZ,
            "EmotionalStressTemp": emotional_Temp,
            "EmotionalStressEDA": emotional_EDA,
            "CognitiveStressAccY": cognitive_AccY,
            "CognitiveStressAccX": cognitive_AccX,
            "CognitiveStressAccZ": cognitive_AccZ,
            "CognitiveStressTemp": cognitive_Temp,
            "CognitiveStressEDA": cognitive_EDA,
        }

        for i, j in enumerate(Label):
            for k in temp_dict1.keys():
                temp_dict2[j + k].append(temp_dict1[k][i])

        # Check the equality of recorded values for each category
        assert len(temp_dict2["RelaxAccX"]) == len(temp_dict2["RelaxEDA"])
        assert len(temp_dict2["PhysicalStressAccX"]) == len(
            temp_dict2["PhysicalStressAccZ"]
        )
        assert len(temp_dict2["CognitiveStressAccX"]) == len(
            temp_dict2["CognitiveStressEDA"]
        )
        assert len(temp_dict2["EmotionalStressAccX"]) == len(
            temp_dict2["EmotionalStressTemp"]
        )

        for i in AccTempEDA_attributes_dict.keys():
            AccTempEDA_attributes_dict[i].append(len(temp_dict2[i + "AccX"]))

        for i in AccTempEDA.keys():
            for j in temp_dict1.keys():
                AccTempEDA[i][j].append(temp_dict2[i + j])

    return AccTempEDA, AccTempEDA_attributes_dict


def necessary_variables_app():
    """
    Define necessary variables and attributes for the application.

    Returns:
        tuple: A tuple containing various dictionaries and lists:
            - SPO2HR_target_size: Dictionary specifying the target size for SPO2HR data for each category.
            - AccTempEDA_target_size: Dictionary specifying the target size for AccTempEDA data for each category.
            - SPO2HR_attributes: List of attributes for SPO2HR data.
            - AccTempEDA_attributes: List of attributes for AccTempEDA data.
            - categories: List of stress categories.
            - attributes_dict: Dictionary containing lists of SPO2HR and AccTempEDA attributes.
            - relax_indices: Dictionary mapping indices for relax category at 8Hz.
            - phy_emo_cog_indices: Dictionary mapping indices for physical, emotional, and cognitive categories at 8Hz.
            - all_attributes: Dictionary mapping indices to all available attributes.
    """

    SPO2HR_target_size = {
        "Relax": 1200,
        "PhysicalStress": 300,
        "EmotionalStress": 300,
        "CognitiveStress": 300,
    }

    AccTempEDA_target_size = {
        "Relax": 9600,
        "PhysicalStress": 2400,
        "EmotionalStress": 3000,
        "CognitiveStress": 2400,
    }

    SPO2HR_attributes = ["Spo2", "HeartRate"]
    AccTempEDA_attributes = ["AccX", "AccY", "AccZ", "Temp", "EDA"]
    categories = ["Relax", "PhysicalStress", "EmotionalStress", "CognitiveStress"]
    attributes_dict = {
        "SPO2HR_attributes": SPO2HR_attributes,
        "AccTempEDA_attributes": AccTempEDA_attributes,
    }

    relax_indices = {
        (i * 8): j for i, j in enumerate(((i + 1) * 8) - 1 for i in range(1200))
    }
    phy_emo_cog_indices = {
        (i * 8): j for i, j in enumerate(((i + 1) * 8) - 1 for i in range(300))
    }

    all_attributes = {
        i: j
        for i, j in enumerate(
            ["SpO2", "HeartRate", "AccX", "AccY", "AccZ", "Temp", "EDA"]
        )
    }

    return (
        SPO2HR_target_size,
        AccTempEDA_target_size,
        SPO2HR_attributes,
        AccTempEDA_attributes,
        categories,
        attributes_dict,
        relax_indices,
        phy_emo_cog_indices,
        all_attributes,
    )


def resize_data_to_uniform_lengths_app(
    total_subject_num,
    categories,
    attributes_dict,
    SPO2HR_target_size,
    SPO2HR,
    AccTempEDA_target_size,
    AccTempEDA,
):
    """
    Resize data for each attribute in SPO2HR and AccTempEDA to a uniform length.

    Parameters:
    - total_subject_num (int): Total number of subjects.
    - categories (list): List of stress categories (e.g., ['Relax', 'CognitiveStress', 'PhysicalStress', 'EmotionalStress']).
    - attributes_dict (dict): Dictionary containing attribute types ('SPO2HR_attributes', 'AccTempEDA_attributes').
    - SPO2HR_target_size (dict): Target size for each stress category in SPO2HR.
    - SPO2HR (dict): Dictionary containing SPO2HR data for each stress category and attribute.
    - AccTempEDA_target_size (dict): Target size for each stress category in AccTempEDA.
    - AccTempEDA (dict): Dictionary containing AccTempEDA data for each stress category and attribute.

    Returns:
    - SPO2HR_temp (dict): Resized SPO2HR data.
    - AccTempEDA_temp (dict): Resized AccTempEDA data.
    """

    # Create deep copies of the input data to avoid modifying the original data
    SPO2HR_temp = copy.deepcopy(SPO2HR)
    AccTempEDA_temp = copy.deepcopy(AccTempEDA)

    # Iterate over stress categories
    for Class in categories:
        # Iterate over attribute types ('SPO2HR_attributes', 'AccTempEDA_attributes')
        for attributes_dict_key in attributes_dict.keys():
            target_attributes = attributes_dict_key

            # Iterate over specific attributes
            for attribute in attributes_dict[attributes_dict_key]:
                if target_attributes == "SPO2HR_attributes":
                    target_size = SPO2HR_target_size[Class]

                    # Iterate over subject numbers
                    for subject_number in range(total_subject_num):
                        temp_list = SPO2HR_temp[Class][attribute][subject_number]
                        offset = len(temp_list) - target_size

                        # Adjust the list length to match the target size
                        if offset > 0:
                            del temp_list[-offset:]
                        elif offset < 0:
                            last_elmt = temp_list[-1]
                            for i in range(-offset):
                                temp_list.append(last_elmt)

                elif target_attributes == "AccTempEDA_attributes":
                    target_size = AccTempEDA_target_size[Class]

                    # Iterate over subject indices
                    for index in range(total_subject_num):
                        temp_list = AccTempEDA_temp[Class][attribute][index]
                        offset = len(temp_list) - target_size

                        # Adjust the list length to match the target size
                        if offset > 0:
                            del temp_list[-offset:]
                        elif offset < 0:
                            last_elmt = temp_list[-1]
                            for i in range(-offset):
                                temp_list.append(last_elmt)

    return SPO2HR_temp, AccTempEDA_temp


def sanity_check_2_and_DownSamplingAccTempEDA_app(
    total_subject_num,
    categories,
    attributes_dict,
    SPO2HR_target_size,
    SPO2HR,
    AccTempEDA_target_size,
    AccTempEDA,
    relax_indices,
    phy_emo_cog_indices,
):
    """
    Perform a sanity check and downsample AccTempEDA data for each attribute.

    Parameters:
    - total_subject_num (int): Total number of subjects.
    - categories (list): List of stress categories (e.g., ['Relax', 'CognitiveStress', 'PhysicalStress', 'EmotionalStress']).
    - attributes_dict (dict): Dictionary containing attribute types ('SPO2HR_attributes', 'AccTempEDA_attributes').
    - SPO2HR_target_size (dict): Target size for each stress category in SPO2HR.
    - SPO2HR (dict): Dictionary containing SPO2HR data for each stress category and attribute.
    - AccTempEDA_target_size (dict): Target size for each stress category in AccTempEDA.
    - AccTempEDA (dict): Dictionary containing AccTempEDA data for each stress category and attribute.
    - relax_indices (dict): Dictionary for downsampling the AccTempEDA values sampled at 8Hz during Relax.
    - phy_emo_cog_indices (dict): Dictionary for downsampling the AccTempEDA values sampled at 8Hz during Physical, Emotional, and Cognitive stresses.

    Returns:
    - AccTempEDA (dict): Modified AccTempEDA data after sanity check and downsampling.
    """

    # Iterate over stress categories
    for Class in categories:
        # Iterate over attribute types ('SPO2HR_attributes', 'AccTempEDA_attributes')
        for attributes_dict_key in attributes_dict.keys():
            target_file = attributes_dict_key

            # Iterate over specific attributes
            for parameter in attributes_dict[attributes_dict_key]:
                if target_file == "SPO2HR_attributes":
                    target_size = SPO2HR_target_size[Class]

                    # Iterate over subject indices
                    for index in range(total_subject_num):
                        temp_list = SPO2HR[Class][parameter][index]
                        temp_list2 = SPO2HR[Class][parameter][-index]

                        # Perform sanity check by comparing lengths of two instances
                        assert len(temp_list) == len(temp_list2)

                elif target_file == "AccTempEDA_attributes":
                    target_size = AccTempEDA_target_size[Class]

                    # Iterate over subject indices
                    for index in range(total_subject_num):
                        temp_list = AccTempEDA[Class][parameter][index]
                        temp_list2 = AccTempEDA[Class][parameter][-index]

                        # Check if the stress category is 'Relax' and the length matches the target size
                        if Class == "Relax" and len(temp_list) == target_size:
                            holding_list = []

                            # Downsample the AccTempEDA values sampled at 8Hz during Relax
                            for key in relax_indices.keys():
                                temp_value = (
                                    sum(temp_list[key : relax_indices[key]])
                                ) / 8
                                holding_list.append(temp_value)

                            AccTempEDA[Class][parameter][index] = holding_list

                        # Check if the stress category is not 'Relax' and the length matches the target size
                        elif Class != "Relax" and len(temp_list) == target_size:
                            holding_list = []

                            # Downsample the AccTempEDA values sampled at 8Hz during Physical, Emotional, and Cognitive stresses
                            for key in phy_emo_cog_indices.keys():
                                temp_value = (
                                    sum(temp_list[key : phy_emo_cog_indices[key]])
                                ) / 8
                                holding_list.append(temp_value)

                            AccTempEDA[Class][parameter][index] = holding_list

    return AccTempEDA


def get_data_dict_app(
    total_subject_num, categories, attributes_dict, SPO2HR, AccTempEDA
):
    """
    This function orgainises the extracted data for easy represention

    Parameters:
    - total_subject_num (int): specifies the total subject number
    - categories (list): a list of categorry names
    - attributes_dict (dict): a dictionary containing all the attributes of both SpO2HR.csv and AccTempEDA.csv files
    - SPO2HR_resized (dict): dictionary containing resized values for Spo2 and HeartRate
    - AccTemEDA_downSampled (dict): dictionary containing resized and downsampled values for Acc(X-Z), Temp, EDA

    RETURNS:
    - ALL_DATA_DICT: a dictionary with integers as keys and numpy arrays as keys
                  first 20 keys: the extracted attributes labelled Relax.
                  second 20 keys: the extracted attributes labelled PhysicalStress
                  third 20 keys: the extracted attributes labelled EmotionalStress
                  fourth 20 keys: the extracted attributes labelled CognitiveStress
    """
    DATA = {
        "Relax": {i + 1: [] for i in range(total_subject_num)},
        "PhysicalStress": {i + 1: [] for i in range(total_subject_num)},
        "EmotionalStress": {i + 1: [] for i in range(total_subject_num)},
        "CognitiveStress": {i + 1: [] for i in range(total_subject_num)},
    }

    DATA_VSTACKS = {
        "Relax": [],
        "PhysicalStress": [],
        "EmotionalStress": [],
        "CognitiveStress": [],
    }

    for (
        Class
    ) in categories:  # Relax', 'CognitiveStress', 'PhysicalStress', 'EmotionalStress'
        for i in range(total_subject_num):
            for (
                attributes_dict_key
            ) in attributes_dict.keys():  # SPO2HR_parameters, AccTempEDA_parameters
                target_file = attributes_dict_key
                for parameter in attributes_dict[
                    attributes_dict_key
                ]:  # 'Spo2', 'HeartRate', ||| 'AccX', 'AccY', 'AccZ', 'Temp', 'EDA'
                    if target_file == "SPO2HR_attributes":
                        DATA[Class][i + 1].extend(SPO2HR[Class][parameter][i])
                    elif target_file == "AccTempEDA_attributes":
                        DATA[Class][i + 1].extend(AccTempEDA[Class][parameter][i])
            if Class == "Relax":
                DATA[Class][i + 1] = np.array(DATA[Class][i + 1]).reshape(
                    7, 1200
                )  # this first 'joins' all samples into a long 1-D array and then reshapes into a 2-D array

                # the total relax of shape (7,1200) would be broken vertically into 4 to give (7,300) compatible samples with the other classes
                # a random number is used to select one out of the 4 samples with which to work with. This ensures our traind dataset is balanced and makes it more robust
                # nth_sample = np.random.randint(0,4)
                # DATA[Class][i + 1] = np.array(np.hsplit(DATA[Class][i + 1], 4))
                # """
                # RUN THE FOLLOWING CODE TO UNDERSTAND THE WORKING OF THE THREE LINES ABOVE
                # num = np.random.randint(0,4)
                # a = np.array(np.hsplit((np.arange(84).reshape(7,12)), 4))[num]
                # print(a)
                # """
            else:
                DATA[Class][i + 1] = np.array(DATA[Class][i + 1]).reshape(7, 300)
            # break
        # break

    DATA_VSTACKS["Relax"] = np.vstack(
        [DATA["Relax"][i + 1] for i in range(total_subject_num)]
    ).reshape(
        total_subject_num * 4, 7, 300
    )  # all the 4 Relax stages used. So first 80 samples are relax
    DATA_VSTACKS["PhysicalStress"] = np.vstack(
        [DATA["PhysicalStress"][i + 1] for i in range(total_subject_num)]
    ).reshape(total_subject_num, 7, 300)
    DATA_VSTACKS["EmotionalStress"] = np.vstack(
        [DATA["EmotionalStress"][i + 1] for i in range(total_subject_num)]
    ).reshape(total_subject_num, 7, 300)
    DATA_VSTACKS["CognitiveStress"] = np.vstack(
        [DATA["CognitiveStress"][i + 1] for i in range(total_subject_num)]
    ).reshape(total_subject_num, 7, 300)

    # """
    # RUN THE FOLLOWING CODE TO UNDERSTAND THE WORKING OF THE FOUR LINES ABOVE
    # DATA_VSTACKS_physicalStress = np.array(np.hsplit((np.arange(84).reshape(7,12)), 4)) # this has the data of 4 subjects with recording lenght of 12, like 1200 :)
    # subject1 = DATA_VSTACKS_physical[0]
    # subject2 = DATA_VSTACKS_physical[1]
    # subject3 = DATA_VSTACKS_physical[2]
    # subject4 = DATA_VSTACKS_physical[3]
    # stack = np.vstack([subject1 ,subject2 ,subject3 ,subject4 ]).reshape(4,7,3) # 4 for number of subjects, 7 for channels, and 3(like 300) for the 5-minute recording
    # print(stack)
    # """

    #  THE ORDER OF STACKINF IS IMPORTANT FOR DETERMINING THE LABELS LABELS. TAKE NOTE
    ALL_DATA = np.vstack(
        [
            DATA_VSTACKS["Relax"],
            DATA_VSTACKS["PhysicalStress"],
            DATA_VSTACKS["EmotionalStress"],
            DATA_VSTACKS["CognitiveStress"],
        ]
    )

    ALL_DATA_DICT = {i: j for i, j in enumerate(ALL_DATA)}
    # """
    # nOTE!!!
    # IF ALL DATASET IS USED, FIRST 80 KEYS OF ALL_DATA_DICT ARE OF CLASS 'Relax'
    # NEXT 20 OF CLASS 'PhysicalStress', then 'EmotionalStress' and 'CognitiveStress'
    # By this reasoning, value at key 0 with shape (7, 300) is entirely from the 'first' samples subject
    # The 7-rows are (Spo2, HeartRate, AccX, AccY, AccY, Temp, EDA) extracted in the relaxed state.
    # """

    return ALL_DATA_DICT


def get_s3_bucket_files(bucket_name):
    """
    Retrieve a list of file keys (names) from an Amazon S3 bucket.

    Parameters:
    - bucket_name (str): The name of the Amazon S3 bucket.

    Returns:
    - List[str]: A list of file keys (names) in the specified S3 bucket.
    """
    s3 = boto3.resource("s3")

    # Access the specified S3 bucket
    bucket = s3.Bucket(bucket_name)

    # List to store file keys
    file_keys = []

    # Iterate through objects in the bucket and append their keys to the list
    for obj in bucket.objects.all():
        file_keys.append(obj.key)

    return file_keys


def get_s3_bucket_tagged_files(
    bucket_name="physiologicalsignalsbucket", sample_window=100, degree_of_overlap=0.5
):
    """
    Retrieve a list of file keys (names) from an Amazon S3 bucket based on specified tags.

    Parameters:
    - bucket_name (str): The name of the Amazon S3 bucket.
    - sample_window (int): The sample window size used as a tag for filtering files.
    - degree_of_overlap (float): The degree of overlap used as a tag for filtering files.

    Returns:
    - List[str] or None: A list of file keys (names) in the specified S3 bucket that match the provided tags.
                         Returns None if no matching files are found.
    """
    s3 = boto3.resource("s3")
    client = boto3.client("s3")

    # Target tags to filter files
    target = {"window": str(sample_window), "overlap": str(degree_of_overlap)}

    # Access the specified S3 bucket
    bucket = s3.Bucket(bucket_name)

    # Lists to store file keys
    buckets = []
    all_buckets = []

    # Iterate through objects in the bucket
    for obj in bucket.objects.all():
        all_buckets.append(obj.key)

        # Retrieve tags for the current object
        response = client.get_object_tagging(Bucket=bucket_name, Key=obj.key)
        file_tags = response["TagSet"]  # a dict

        # Counter to track matching tags
        counter = 0

        # Check if tags match the target tags
        for element in file_tags:
            try:
                if element["Key"] in target.keys():
                    if element["Value"] == target[element["Key"]]:
                        counter += 1
                    if counter >= 2:
                        buckets.append(obj.key)
            except Exception as e:
                # Handle errors when getting tags
                print(f"Got an error in getting tags of file. Error: {e}")

    # Display a warning if no matching files are found
    if len(buckets) == 0:
        print(
            "Found no compatible model for selected sample window and degree of overlap."
        )
        print(
            "All models are displayed below. Either change the selection or train a model with specifications."
        )
        return None

    return buckets


def download_s3_file(
    s3_file_path,
    bucket_name="physiologicalsignalsbucket",
    model_local_path="./temp/models/downloaded_model.h5",
):
    """
    Download a file from an Amazon S3 bucket to a local path.

    Parameters:
    - s3_file_path (str): The path of the file in the Amazon S3 bucket.
    - bucket_name (str): The name of the Amazon S3 bucket.
    - model_local_path (str): The local path where the downloaded file should be saved.

    Returns:
    - str: The local path where the file has been downloaded.
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    # Download the file from S3 to the local path
    s3.download_file(bucket_name, s3_file_path, model_local_path)

    # Return the local path of the downloaded file
    return model_local_path


def upload_file_to_s3(
    file_path="./temp/models/model.h5",
    bucket_name="physiologicalsignalsbucket",
    object_name=None,
    window=100,
    overlap=0.5,
):
    """
    Upload a file to an Amazon S3 bucket with specified tags.

    Parameters:
    - file_path (str): The local path of the file to be uploaded.
    - bucket_name (str): The name of the Amazon S3 bucket.
    - object_name (str): The name of the object in the S3 bucket. If None, the file name is used.
    - window (int): The sample window parameter for tagging (default: 100).
    - overlap (float): The degree of overlap parameter for tagging (default: 0.5).

    Returns:
    - None: The function does not return any value but prints success or error messages.
    """
    s3_client = boto3.client("s3")

    # Use the file name as the object name if not specified
    if object_name is None:
        object_name = file_path.split("/")[-1]

    try:
        # Create tags for the file
        tags = f"window={str(window)}&overlap={str(overlap)}"

        # Upload the file to S3 with specified tags
        s3_client.upload_file(
            file_path, bucket_name, f"{object_name}.h5", ExtraArgs={"Tagging": tags}
        )

        print(f"File uploaded successfully to S3 bucket: {bucket_name}")
    except Exception as e:
        print(f"Error uploading file to S3 bucket: {e}")
