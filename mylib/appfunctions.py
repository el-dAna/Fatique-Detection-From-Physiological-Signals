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
    # https://physionet.org/content/noneeg/1.0.0/


@dataclass
class DATA_VARIABLES:
    Relax = 5
    PhysicalStress = 6
    MiniCognitiveStress = 40
    CognitiveStress = 5
    EmotionalStress = 5
    Total_time_minutes = (
        Relax * 4 + PhysicalStress + CognitiveStress + EmotionalStress
    ) + (MiniCognitiveStress / 60)
    Total_time_seconds = Total_time_minutes * 60
    freq_eda_files = 18230 / Total_time_seconds
    # freq_spo2_files =


def write_expandable_text_app(
    title, detailed_description, img_path=False, variable=False
):
    """
    Displays a callapsed image/variable(dict, list, dataframe)
    str title: Text to display on collapsed bar
    str detailed_description: detailed description to display when uncollapsed
    str img_path: path of image to show when uncollapsed
    str variable: varialbe to display(if any)
    """
    with st.expander(title):
        st.write(detailed_description)
        if img_path:
            st.image(img_path)
        if variable:
            st.write(variable)


def display_collapsed_dict_app(dictionary):
    with st.expander("Dictionary"):
        st.write(dictionary)


def upload_files(from_s3=False):
    """
    Function to upload files using Streamlit file_uploader.

    Returns:
    - dict or []: dict of file names and uploaded file object from the streamlit file_uploader or [] if no files are uploaded.
    """

    file_names = []
    uploaded_files_dict = {}
    if from_s3:
        for i in range(20):
            file_names.append("Subject" + str(i + 1))
        uploaded_files = st.multiselect(
            "Select subjects whose data you want to load", file_names
        )

    else:
        uploaded_files = st.file_uploader(
            "Upload files", type="csv", accept_multiple_files=True
        )

    if uploaded_files:
        for file in uploaded_files:
            if from_s3:
                bucket_name = "physiologicalsignals"
                folder_name = "HealthySubjectsBiosignalsDataSet"
                file_path1 = f"{bucket_name}/{folder_name}/{file}/{file}SpO2HR.csv"
                file_path2 = f"{bucket_name}/{folder_name}/{file}/{file}AccTempEDA.csv"

                # Read the CSV file
                # data = pd.read_csv(s3_object)
                data1 = pd.read_csv(f"s3://{file_path1}")
                data2 = pd.read_csv(f"s3://{file_path2}")

                uploaded_files_dict[f"{file}SpO2HR.csv"] = data1
                uploaded_files_dict[f"{file}AccTempEDA.csv"] = data2

            else:
                file_names.append(file.name)
                uploaded_files_dict[file.name] = pd.read_csv(file)

                if len(file_names) % 2 != 0:
                    st.error("Please upload an even number of files.")
                    return None

        return uploaded_files_dict

    return []


def read_files(uploaded_files_dict):
    """
    Function to read CSV files from and plot data based on the selected files.

    Args:
    - uploaded_filees_dict (dict): dict of file names as values and uploaded file object from calling st.file_uploader().

    Returns:
    - None: List of DataFrames or None if an error occurs.
    """

    # selected_file = st.selectbox("Select an option:", uploaded_files_dict.keys())

    selected_files = st.multiselect("Select subject data", uploaded_files_dict.keys())
    if selected_files:
        # Create columns for plots
        graph_cols = st.columns(int(len(selected_files)))
        pandas_cols = st.columns(int(len(selected_files)))

        for i, file in enumerate(selected_files):
            with pandas_cols[i]:
                # Create pandas dataframe
                dataframe = uploaded_files_dict[file]
                st.write(dataframe)
                st.write("adfa", len(dataframe["Label"]))
                time_steps = range(len(dataframe["Second"]))
                dataframe["Second_modified"] = time_steps
            with graph_cols[i]:
                if "EDA" in file:
                    dataframe.plot(
                        x="Second_modified", y=["AccX", "AccY", "AccZ", "Temp", "EDA"]
                    )
                    plt.xlabel("Seconds")
                    plt.ylabel("Recorded value")
                    plt.title(f"Plot of recorded signals of {file}")
                    plot_vertical_lines(plot=plt, freq=8.286363636363637)
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


def plot_vertical_lines(plot, freq=1):
    plot.axvline(x=(5 * 60) * freq, color="b", label="axvline - full height")  # relax
    plot.axvline(
        x=(5 * 60 + 6 * 60) * freq, color="r", label="axvline - full height"
    )  # physical stress
    plot.axvline(
        x=(5 * 60 + 6 * 60 + 5 * 60) * freq, color="b", label="axvline - full height"
    )  # relax
    plot.axvline(
        x=(5 * 60 + 6 * 60 + 5 * 60 + 40) * freq,
        color="g",
        label="axvline - full height",
    )  # minicognitive
    plot.axvline(
        x=(5 * 60 + 6 * 60 + 5 * 60 + 40 + 5 * 60) * freq,
        color="g",
        label="axvline - full height",
    )  # cognitive
    plot.axvline(
        x=(5 * 60 + 6 * 60 + 5 * 60 + 40 + 5 * 60 + 5 * 60) * freq,
        color="b",
        label="axvline - full height",
    )  # relax
    plot.axvline(
        x=(5 * 60 + 6 * 60 + 5 * 60 + 40 + 5 * 60 + 5 * 60 + 5 * 60) * freq,
        color="y",
        label="axvline - full height",
    )  # emotional
    plot.axvline(
        x=(5 * 60 + 6 * 60 + 5 * 60 + 40 + 5 * 60 + 5 * 60 + 5 * 60 + 5 * 60) * freq,
        color="b",
        label="axvline - full height",
    )  # relax


def get_numerical_labels(dataframe):
    return dataframe


def group_dataframe_by(dataframe, column_name="Label"):
    session_grp = dataframe.groupby(column_name)
    return session_grp


def SortSPO2HR_app(uploaded_files_dict, uploaded_spo2_files):
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

        # Extracting the SpO2 and heartRate columns of each subject
        spo2 = np.array(list(data2["SpO2"]))
        HeartRate = np.array(list(data2["HeartRate"]))
        labels = list(data2["Label"])  # the labels are strings!!
        labels_dict = {j: i for i, j in enumerate(set(labels))}

        # """
        # Empty list initialisation to store the values of each category.
        # For example, there are relax, PhysicalStress, EmotionalStress and CognitiveStress
        # in the extracted SpO2 column, so we want to extract the measured voltages
        # and append to the appropriate empty list
        # """
        relax_spo2, relax_HeartRate = [], []
        cognitive_spo2, cognitive_HeartRate = [], []
        physical_spo2, physical_HeartRate = [], []
        emotional_spo2, emotional_HeartRate = [], []

        index = 0

        for j, i in enumerate(labels):
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

        # """
        # Since both SpO2 and HeartRate were measured at the same frequency[1Hz] and time, then the number
        # of recorded values for each catogory should be equal. The following assetions check that
        # """
        assert len(relax_spo2) == len(relax_HeartRate)
        assert len(physical_spo2) == len(physical_HeartRate)
        assert len(emotional_spo2) == len(emotional_HeartRate)
        assert len(cognitive_spo2) == len(cognitive_HeartRate)
        assert len(relax_spo2) + len(physical_spo2) + len(emotional_spo2) + len(
            cognitive_spo2
        ) == len(labels)

        # """
        # This dictionary stores the length of each category from each subject
        # For example, the Relax key stores the total recording time for relax for each subject.
        # SPO2HR_attributes_dict['Relax'][0] gives subject1 total relax time which equals 1203.
        # 1203 interpretation.
        # There were 4 relax stages of 5mins each = 5*60*4  = 1200
        # """
        SPO2HR_attributes_dict["Relax"].append(len(relax_spo2))
        SPO2HR_attributes_dict["PhysicalStress"].append(len(physical_spo2))
        SPO2HR_attributes_dict["CognitiveStress"].append(len(cognitive_spo2))
        SPO2HR_attributes_dict["EmotionalStress"].append(len(emotional_spo2))

        # time = np.arange(len(relax_spo2))
        # plt.plot(time, relax_HeartRate )
        # plt.show()

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
        # break

    return SPO2HR, SPO2HR_attributes_dict


def SortAccTempEDA_app(uploaded_files_dict, uploaded_tempEda_files):
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

        # Declaring empty list variables to store extracts for specific categories
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

        # print(f'Subject: {subject}')
        # print(f'Relax: {len(temp_dict2["RelaxAccX"])}')
        # print(f'Cognitive: {len(temp_dict2["CognitiveStressAccX"])}')
        # print(f'Physical: {len(temp_dict2["PhysicalStressAccX"])}')
        # print(f'Emotional: {len(temp_dict2["EmotionalStressAccX"])} \n')

        for i in AccTempEDA_attributes_dict.keys():
            AccTempEDA_attributes_dict[i].append(len(temp_dict2[i + "AccX"]))

        for i in AccTempEDA.keys():
            for j in temp_dict1.keys():
                AccTempEDA[i][j].append(temp_dict2[i + j])

    return AccTempEDA, AccTempEDA_attributes_dict


def necessary_variables_app():
    """
    This fucntion returns major variables to be used in the training file.
    Some variables are declared and initiated within this function.
    This is done for better organisation and debugging.
    """
    # here were 4 relax stages each of 5 mins each at 1Hz = 4*5*60 = 1200, so the standard total recoding time for relax must equal 1200
    # There was 1 stage for the remaing 3 categories, 5min each at 1Hz = 1*5*60 = 300
    SPO2HR_target_size = {
        "Relax": 1200,
        "PhysicalStress": 300,
        "EmotionalStress": 300,
        "CognitiveStress": 300,
    }  # 300 set for EmotionalStress cos most

    # There were 4 relax stages each of 5 mins each at 8Hz = 4*5*60*8 = 9600, so the standard total recoding time for relax at 8Hz must equal 9600
    # There was 1 stage for the remaing 3 categories, 5min each at 8Hz = 1*5*60*8 = 2400
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
        [i * 8 for i in range(1200)][i]: j
        for i, j in enumerate([((i + 1) * 8) - 1 for i in range(1200)])
    }
    phy_emo_cog_indices = {
        [i * 8 for i in range(300)][i]: j
        for i, j in enumerate([((i + 1) * 8) - 1 for i in range(300)])
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
    This function resizes the varying recorded total times for the various categories to the targetted recording time.
    For example, total relax recording time for Subject1 = 1203, but the targetted = 1200. So this function removes the excesses or appends the last recorded values

    INPUTS:
    total_subject_num: (int) the total suject number
    categories: a list -> contains the category names
    attributes_dict: a dict -> contains the attributes[Spo2, HeartRate, Acc(X-Z), Temp, EDA] of the dataset
    SPO2HR_target_size: a dict -> contains the theoritical lengths(number of recorded values) that each category should be in the SPO2HR.csv folder. 1Hz
    SPO2HR: A dictionary of the categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
          The value of each key is a dictionary with Spo2 and HeartRate as keys.
          The values of each key is a list containing the measured voltages from a subject

    AccTempEDA_target_size: a dict -> contains the theorical lengths(number of recorded values) that each category should be in the AccTempEDA.csv folder. 8Hz
    AccTempEDA: a dictionary with the categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
                The value of each first key is a dictionary with attributes(AccZ, AccY, AccX, Temp, EDA) as keys
                The value of each sencond key is a list.
                The list contains the extracted values of attributes for each subject
                AccTempEDA['Relax']['AccZ'][0] contains the extracted relax values of AccZ column of subject 1

    RETURNS:
    SPO2HR: A dictionary of the RESIZED TO UNIFORM LENGTH categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
          The value of each key is a dictionary with Spo2 and HeartRate as keys.
          The values of each key is a list containing the measured voltages from a subject

    AccTempEDA: a dictionary with the RESIZED TO UNIFORM LENGHT categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
                The value of each first key is a dictionary with attributes(AccZ, AccY, AccX, Temp, EDA) as keys
                The value of each sencond key is a list.
                The list contains the extracted values of attributes for each subject
                AccTempEDA['Relax']['AccZ'][0] contains the extracted relax values of AccZ column of subject 1

    """
    SPO2HR_temp = copy.deepcopy(SPO2HR)
    AccTempEDA_temp = copy.deepcopy(AccTempEDA)

    for (
        Class
    ) in categories:  # Relax', 'CognitiveStress', 'PhysicalStress', 'EmotionalStress'
        for (
            attributes_dict_key
        ) in attributes_dict.keys():  # SPO2HR_parameters, AccTempEDA_parameters
            target_attributes = attributes_dict_key
            # print((attributes_dict_key))
            for attribute in attributes_dict[
                attributes_dict_key
            ]:  # 'Spo2', 'HeartRate', ||| 'AccX', 'AccY', 'AccZ', 'Temp', 'EDA'
                if target_attributes == "SPO2HR_attributes":
                    # print("IN SPO2HR NOW!!")
                    target_size = SPO2HR_target_size[Class]
                    for subject_number in range(total_subject_num):
                        temp_list = SPO2HR_temp[Class][attribute][subject_number]
                        offset = len(temp_list) - target_size
                        if offset > 0:
                            del temp_list[-offset:]
                            assert len(temp_list) == target_size
                        elif offset < 0:
                            last_elmt = temp_list[-1]
                            for i in range(-offset):
                                temp_list.append(last_elmt)
                            assert len(temp_list) == target_size
                        elif offset == 0:
                            assert len(temp_list) == target_size
                            # pass

                elif target_attributes == "AccTempEDA_attributes":
                    # print("IN AccTempEDA NOW!!")
                    target_size = AccTempEDA_target_size[Class]
                    for index in range(total_subject_num):
                        temp_list = AccTempEDA_temp[Class][attribute][index]
                        offset = len(temp_list) - target_size
                        if offset > 0:
                            del temp_list[-offset:]
                            assert len(temp_list) == target_size
                        elif offset < 0:
                            last_elmt = temp_list[-1]
                            for i in range(-offset):
                                temp_list.append(last_elmt)
                            assert len(temp_list) == target_size
                        elif offset == 0:
                            assert len(temp_list) == target_size
                            # pass
                    # break
                # break
            # break
        # break
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
    This function checks the accuracy of the preprocessed data so far by comparing the preprocessed values with the originals.
    In order not to define a second function, the 8Hz Acc(X-Z), Temp and EDA lenghts were downsampled to match the 1Hz sampling of Spo2 and HeartRate

    INPUTS:
    total_subject_num: (int) the total suject number
    categories: a list -> contains the category names
    attributes_dict: a dict -> contains the attributes[Spo2, HeartRate, Acc(X-Z), Temp, EDA] of the dataset
    SPO2HR_target_size: a dict -> contains the theoritical lengths(number of recorded values) that each category should be in the SPO2HR.csv folder. 1Hz
    SPO2HR: A dictionary of the categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
          The value of each key is a dictionary with Spo2 and HeartRate as keys.
          The values of each key is a list containing the measured voltages from a subject

    AccTempEDA_target_size: a dict -> contains the theorical lengths(number of recorded values) that each category should be in the AccTempEDA.csv folder. 8Hz
    AccTempEDA: a dictionary with the categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
                The value of each first key is a dictionary with attributes(AccZ, AccY, AccX, Temp, EDA) as keys
                The value of each sencond key is a list.
                The list contains the extracted values of attributes for each subject
                AccTempEDA['Relax']['AccZ'][0] contains the extracted relax values of AccZ column of subject 1
    relax_indices: a dict -> contains the indices of values of relax. This allows easy sampling by direct referencing.
    phy_emo_cog_indices: a dict -> contains the indices of values of PhysicalStress, EmotionalStress and Cognitive Stress for easy sampling by referencing


    RETURNS:
    AccTempEDA: a dictionary with the DOWNSAMPLED categories(Relax, PhysicalStress, CognitiveStress, EmotionalStress) as keys
                The value of each first key is a dictionary with attributes(AccZ, AccY, AccX, Temp, EDA) as keys
                The value of each sencond key is a list.
                The list contains the extracted values of attributes for each subject
                AccTempEDA['Relax']['AccZ'][0] contains the extracted relax values of AccZ column of subject 1


    """
    for (
        Class
    ) in categories:  # Relax', 'CognitiveStress', 'PhysicalStress', 'EmotionalStress'
        for (
            attributes_dict_key
        ) in attributes_dict.keys():  # SPO2HR_parameters, AccTempEDA_parameters
            target_file = attributes_dict_key
            for parameter in attributes_dict[
                attributes_dict_key
            ]:  # 'Spo2', 'HeartRate', ||| 'AccX', 'AccY', 'AccZ', 'Temp', 'EDA'
                if target_file == "SPO2HR_attributes":
                    # print("IN SPO2HR NOW!!")
                    target_size = SPO2HR_target_size[Class]
                    for index in range(total_subject_num):
                        temp_list = SPO2HR[Class][parameter][index]
                        temp_list2 = SPO2HR[Class][parameter][-index]
                        assert len(temp_list) == len(temp_list2)

                elif target_file == "AccTempEDA_attributes":
                    # print("IN AccTempEDA NOW!!")
                    target_size = AccTempEDA_target_size[Class]
                    for index in range(
                        total_subject_num
                    ):  # use this line for the resizing
                        temp_list = AccTempEDA[Class][parameter][index]
                        ### offset = len(temp_list) - target_size

                        temp_list2 = AccTempEDA[Class][parameter][-index]
                        if Class == "Relax" and len(temp_list) == target_size:
                            holding_list = []
                            for (
                                key
                            ) in (
                                relax_indices.keys()
                            ):  # dict for downsapmling of the AccTempEDA values sampled at 8HZ
                                temp_value = (
                                    sum(temp_list[key : relax_indices[key]])
                                ) / 8
                                holding_list.append(temp_value)
                            AccTempEDA[Class][parameter][index] = holding_list
                        elif Class != "Relax" and len(temp_list) == target_size:
                            holding_list = []
                            for (
                                key
                            ) in (
                                phy_emo_cog_indices.keys()
                            ):  # this dict is same as the relax_indices but shorter in length. This only spans 300 values. The relax is 4 times this one.
                                temp_value = (
                                    sum(temp_list[key : phy_emo_cog_indices[key]])
                                ) / 8
                                holding_list.append(temp_value)
                            AccTempEDA[Class][parameter][index] = holding_list
                        else:
                            print("Passing")
                        # """
                    # break
                # break
            # break
        # break
    return AccTempEDA


def get_data_dict_app(
    total_subject_num, categories, attributes_dict, SPO2HR, AccTempEDA
):
    """
    This function orgainises the extracted data for easy represention

    total_subject_num: int. specifies the total subject number
    categories: a list of categorry names
    attributes_dict: a dictionary containing all the attributes of both SpO2HR.csv and AccTempEDA.csv files
    SPO2HR_resized: dictionary containing resized values for Spo2 and HeartRate
    AccTemEDA_downSampled: dictionary containing resized and downsampled values for Acc(X-Z), Temp, EDA

    RETURNS:
    ALL_DATA_DICT: a dictionary with integers as keys and numpy arrays as keys
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
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    buckets = []
    for obj in bucket.objects.all():
        buckets.append(obj.key)
    return buckets


def download_s3_file(
    s3_file_path,
    bucket_name="physiologicalsignalsbucket",
    model_local_path="./data/models/downloaded_model.h5",
):
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, s3_file_path, model_local_path)


def upload_file_to_s3(
    file_path="./data/models/model.h5",
    bucket_name="physiologicalsignalsbucket",
    object_name=None,
    window=100,
    overlap=0.5,
):
    s3_client = boto3.client("s3")
    if object_name is None:
        object_name = file_path.split("/")[-1]  # Use the file name as the object name
    try:
        tags = f"window={str(window)}&overlap={str(overlap)}"
        s3_client.upload_file(
            file_path, bucket_name, f"{object_name}.h5", ExtraArgs={"Tagging": tags}
        )
        print(f"File uploaded successfully to S3 bucket: {bucket_name}")
    except Exception as e:
        print(f"Error uploading file to S3 bucket: {e}")
