import streamlit as st
import os
import pandas as pd
from tempfile import NamedTemporaryFile

# import boto3
# from io import BytesIO


def upload_files():
    """
    Function to upload files using Streamlit file_uploader.

    Returns:
    - list or None: List of uploaded files or None if no files are uploaded.
    """
    file_names = []
    uploaded_files_dict = {}
    uploaded_files = st.file_uploader(
        "Upload files", type="csv", accept_multiple_files=True
    )

    st.write(uploaded_files)
    if uploaded_files:        
        for file in uploaded_files:
            file_names.append(file.name)
            uploaded_files_dict[file.name] = file
        

            # with NamedTemporaryFile() as temp:
            #     temp.write(file.getvalue())
            #     temp.flush()
            #     file_paths.append(temp.name)
            #     st.write(f"File path: {file_paths}")

        if len(file_names) % 2 != 0:
            st.error("Please upload an even number of files.")
            return None
        return uploaded_files_dict

    return []


def read_files(uploaded_files_dict):
    """
    Function to read CSV files from the given file paths.

    Args:
    - file_paths (list): List of file paths.

    Returns:
    - list or None: List of DataFrames or None if an error occurs.
    """
    
    selected_file = st.selectbox("Select an option:", uploaded_files_dict.keys())
    
    
    selected_file=uploaded_files_dict[selected_file]
    #file_path = os.path.join(tempfile.gettempdir(), selected_file)
    # st.write(f"File path: {selected_file}")
    dataframe = pd.read_csv(selected_file)
    st.write(dataframe)

    # try:
    #     data = pd.read_csv(selected_file)
    #     st.write(data.describe())
    # except FileNotFoundError as e:
    #     st.error(f"Error reading file {selected_file}: {e}")
    #     return None

    # def load_data_from_s3_bucket(bucket_name, object_key, s3_client=boto3.client('s3')):
    #     response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    #     #data = pd.read_csv(response['Body'])

    #     return type(data)


# # AWS credentials (replace with your own credentials)
# aws_access_key_id = "your_access_key_id"
# aws_secret_access_key = "your_secret_access_key"
# aws_region = "your_region"

# # S3 bucket and file details
# bucket_name = "physiologicalsignals"
# file_key = "HealthySubjectsBiosignalsDataSet/Subject1/Subject1AccTempEDA.csv"


# # Function to load data from S3
# def load_data_from_s3():
#     s3 = boto3.client(
#         "s3",
#         aws_access_key_id=aws_access_key_id,
#         aws_secret_access_key=aws_secret_access_key,
#         region_name=aws_region,
#     )
#     try:
#         # Download the file from S3
#         obj = s3.get_object(Bucket=bucket_name, Key=file_key)
#         # data = pd.read_csv(BytesIO(obj['Body'].read()))
#         return obj
#     except Exception as e:
#         st.error(f"Error loading data from S3: {str(e)}")
#         return None
