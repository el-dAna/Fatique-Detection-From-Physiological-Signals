# Fatique-Detection-From-Physiological-Signals

[![Python application test with Github Actions](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/main.yml/badge.svg)](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/main.yml)

[![DockerImage 2 ECR Repo](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/docker_img2ecr.yml/badge.svg)](https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/actions/workflows/docker_img2ecr.yml)



https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/assets/100404513/2dff8d8c-881c-4ec3-8c88-b19b6f4f1c7c


### ABOUT REPO
This is more of an MLOPS orchestration than a useful ML model. This project classifies physiological signals into for stress categories. Users can upload from local environment or from data on s3 bucket.

MAIN DEMONSTRATIONS.
- Makefile for convenient commandline commands 
- CI/CD Pipelines (Git Actions to test, lint and format code)
- CI/CD Pipeline to containerise imaage, tag it as latest, upload to ECS repo and AppRunner takes off from there.
- Tensorflow Model Development
- Custom DataGenerator specifically for this type of data
- Experiment tracking using ClearML instead of Mlflow for more nuanced logging and control
- Model Optimisation using ClearML
- Agent Deployment and Remote Model Training
- Trained Model hosting in s3 bucket
- Subject Data hosted in s3 bucket
- FastApi to get list of run and completed experiments (proof of concept)

## Data
More information on data can be obtained from the official PhysionNet [here](https://physionet.org/content/noneeg/1.0.0/). For a visual representaion visit the Figma visual representation [here](https://www.figma.com/file/qAkiRCvXSZOOgIAfwgJiNY/Tuseb?type=design&node-id=13%3A2&mode=design&t=8ijg5b9MX6MiBvID-1).
To get indepth analysis of the data and how it was 'organised'(no major preprocessing yet) for the model, visit the Data_Preprocessing Page of the Streamlit App.
<div><img width="574" alt="image" src="https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/assets/100404513/c7cce69f-cf71-4ffb-8a3a-47ea2c475b92"></div>


### Model
A tensorflow model accepting variable input data shapes depending in user's selection via the Streamlit UI.
The model is trained and logged on clearml but hosted in s3 under a custom name. 
Users can later load specific trained model to run infererence.

### S3
This is storage for this project. 
Subject data and models are hosted here.
<div><img width="600" alt="image" src="https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/assets/100404513/6a8d6621-6b60-4c58-9c77-5e5eb2709741"></div>



### ClearML
Project runs are tracked here.
Old runs can be cloned and rerun to simulate results.(not available on Mlflow)
Runs and generated models can be compared.
Automatic logging of artifcacts.
Models could be hosted here but I just wanted to use diverse tools.

<div><img width="600" alt="image" src="https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/assets/100404513/b24b1712-63c2-4cbe-9520-a01077dcdda8"></div>



### SUMMARY
Users upload data and directly make inference on data using models from s3. OR
Visualise/Organise the data their way.
Train model based on specification and save in s3 bucket.
Load the model later for inference on data.


### CODE STRUCTURE
Entry is the App.py
    Pages
     - Data_Precessing.py
     - Model_Inference.py
     - Train_Model.py
    These call other modules from other libs.
For an interactive display visit the figma file [here](https://www.figma.com/file/qAkiRCvXSZOOgIAfwgJiNY/Tuseb?type=design&node-id=13%3A2&mode=design&t=8ijg5b9MX6MiBvID-1).
<div><img width="600" alt="image" src="https://github.com/el-dAna/Fatique-Detection-From-Physiological-Signals/assets/100404513/45780569-f974-49b6-ba12-7e8b4e7fad44"></div>


### RUNNING APP (On LINUX)
1. Move to directory of repo
2. Run `python -m venv st_portfolio` to create a virtual environment called st_portfolio
3. Run `source st_portfolio/bin/activate` to activate
4. Run `pip install -r requirements.txt` to install packages
5. Create a ClearML account [here](https://clear.ml/docs/latest/)
6. Follow easy-to-understand steps [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/) to authenticate environment to ClearML server.
7. Run `streamlit run App.py` to run app hosted at port `8501`
Enjoy locally. To have full experence download the dataset from the link and upload files when requested or host on s3 and grant the necessary policies to an IAM role, generate keys, authenticate your environment and that's it!


### BUILDING DOCKER IMAGE LOCALLY
1. Run `docker build -t Image1 .` to build image with tagname Image1 
2. Run `docker image ls`to see images built so far
3. Run `docker run -p 127.0.0.1:8080:8080 Image1` to deploy image locally
4. Run `docker rmi image_id --force` to delete image with ID==image_id 
5. `docker rmi $(docker images -q) --force` to delete ALL images. Images are about 2.9GB so you will need to delete often, except if your storage is an organic brain :).

#`docker stop container`




