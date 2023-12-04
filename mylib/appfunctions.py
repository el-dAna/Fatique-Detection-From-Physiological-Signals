import streamlit as st
from dataclasses import dataclass

# import os
import pandas as pd
import numpy as np

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


def upload_files():
    """
    Function to upload files using Streamlit file_uploader.

    Returns:
    - dict or []: dict of file names and uploaded file object from the streamlit file_uploader or [] if no files are uploaded.
    """

    file_names = []
    uploaded_files_dict = {}
    uploaded_files = st.file_uploader(
        "Upload files", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
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
    # labels = dataframe["Labels"]
    # (dataframe.Labels.values == 'PhysicalStress').argmin()
    # (dataframe.Labels.values == 'CognitiveStress').argmin()
    # (dataframe.Labels.values == 'EmotionalStress').argmin()
    # labels = ['Relax', 'PhysicalStress', 'CognitiveStress', 'EmotionalStress']
    # labels_dict = dict(zip(labels, range(0,4)))
    # dataframe['labels'].map(labels_dict, na_action='ignore')
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
