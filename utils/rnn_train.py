from dataclasses import dataclass
from clearml import Task
import numpy as np
#from typing import Tuple
import tensorflow as tf
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from pprint import pprint
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix #, classification_report
import sys
import s3fs
import boto3
import streamlit as st


from .rnn_model import model
from .common_functions import (
    train_stack,
    predict_stack,
    #adjust_sensitivity,
    window_sampling,
    #PhysioDatagenerator,
    stop_training,
    #schedule_learningRate,
    #plot_learnRate_epoch,
    #plot_loss_accuracy,
)
from .preprocessingfunctions import (
    #SortSPO2HR,
    #SortAccTempEDA,
    #sanity_check_1,
    #necessary_variables,
    #resize_to_uniform_lengths,
    sanity_check_2_and_DownSamplingAccTempEDA,
    get_data_dict,
    plot_varying_recording_time,
    get_variables,
)

from mylib.appfunctions import upload_file_to_s3

tf.keras.backend.clear_session()  # clears internal variables so we start all initiations and assignments afresh


@dataclass
class RNN_TRAIN_DATACLASS:
    BASE_DIR = "./HealthySubjectsBiosignalsDataSet/"
    PATH_TO_SAVED_VARIABLES = "./utils/saved_vars.py"

    print("Running RNN_TRAIN_.... eventhough not called directly")

    (
        WHOLE_DICT,
        CATEGORIES,
        LABELS_TO_NUMBERS_DICT,
        NUMBERS_TO_LABELS_DICT,
    ) = get_variables(PATH_TO_SAVED_VARIABLES)
    # SAVED_CWT_DICT = {i:j/255. for i,j in enumerate(SAVED_CWT_DICT['features'])}

    NUMBER_CLASSES = 4

        
    # Get the current date and time
    clearml_project_name = "portfolioproject"
    current_datetime = str(datetime.datetime.now())
    clearml_task_name= f"task-{current_datetime}"
    model_local_path= str("./temp/models/model.h5")
    bucket_name=str("physiologicalsignalsbucket"),
    model_s3_name= str(f"Model-{current_datetime}")
    




def init_clearml_task(project_name=RNN_TRAIN_DATACLASS.clearml_project_name, task_name=RNN_TRAIN_DATACLASS.clearml_task_name):
    task_name = Task.init(project_name=project_name, task_name=task_name)
    return task_name


def initialise_training_variables(sample_window=100, degree_of_overlap=0.5, WHOLE_DICT=RNN_TRAIN_DATACLASS.WHOLE_DICT,
                                  PERCENT_OF_TRAIN=0.8):
    WINDOW_SAMPLING_DICT = {
        i: j
        for i, j in enumerate(
            window_sampling(WHOLE_DICT, window_size=sample_window, overlap=degree_of_overlap)
        )
    }
    TOTAL_GEN_SAMPLES = len(WINDOW_SAMPLING_DICT.keys())
    SAMPLES_PER_SAMPLE = int(TOTAL_GEN_SAMPLES / len(WHOLE_DICT.keys()))

    RELAX_PROPORTION = 80 * SAMPLES_PER_SAMPLE #there are originally 80 features labeled as relax
    OTHERS_PROPORTION = 20 * SAMPLES_PER_SAMPLE #there are originally 20 features labeled for each of the remaining classes (physicalstress, emotionalstress and cognituvestress)

    TRAIN_RELAX_PROPORTION = int(PERCENT_OF_TRAIN * RELAX_PROPORTION) #how many of the (number of relax sampled to generate a dataset) are used for training 
    TRAIN_OTHERS_PROPORTION = int(PERCENT_OF_TRAIN * OTHERS_PROPORTION) #how many of the (number of other labels sampled to generate a dataset) are used for training 


    TRAIN_FEATURES = train_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=True,
    )
    TOTAL_TRAIN_DATA = len(TRAIN_FEATURES)
    INPUT_FEATURE_SHAPE = TRAIN_FEATURES[0].shape

    TRAIN_LABELS = train_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        #train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=False,
    )
    

    PREDICT_FEATURES = predict_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        #train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=True,
    )
    TOTAL_VAL_DATA = len(PREDICT_FEATURES)

    PREDICT_LABELS = predict_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        #train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=False,
    )

    TRAIN_BATCH_SIZE = int(TOTAL_TRAIN_DATA / 8)  # /8
    assert (
        TOTAL_TRAIN_DATA % TRAIN_BATCH_SIZE == 0
    ), "Ensure that the batch size is perfectly divisible by total_train_data"

    # VAL_BATCH_SIZE = int(TOTAL_VAL_DATA) # /4
    # assert(TOTAL_VAL_DATA % VAL_BATCH_SIZE == 0), "Ensure teh val_batch_size is perfectly divisible by the total_val_data"

    TRAIN_STEPS = int(TOTAL_TRAIN_DATA // TRAIN_BATCH_SIZE)
    # VAL_STEPS = int(TOTAL_VAL_DATA // VAL_BATCH_SIZE)


    return TRAIN_FEATURES, TRAIN_LABELS, TOTAL_TRAIN_DATA, PREDICT_FEATURES, PREDICT_LABELS, TOTAL_VAL_DATA, INPUT_FEATURE_SHAPE, TRAIN_BATCH_SIZE, TRAIN_STEPS



def initialise_train_model(MODEL_INPUT_SHAPE, 
                dp1 = 0.3,
                dp2 = 0.3,
                dp3 = 0.0,
                dp4 = 0.0,
                learning_rate = 0.0002,
                LOSS=tf.keras.losses.Huber(),
                NUMBER_CLASSES=RNN_TRAIN_DATACLASS.NUMBER_CLASSES):

   

    # Callbacks = [stop_training(), schedule_learningRate]
    Callbacks = [stop_training()]

    # params_for_mlflow_log = {
    #     "dp1": RNN_TRAIN_DATACLASS.dp1,
    #     "dp2": RNN_TRAIN_DATACLASS.dp2,
    #     "dp3": RNN_TRAIN_DATACLASS.dp3,
    #     "learning_rate":RNN_TRAIN_DATACLASS.learning_rate
    # }

    model_to_train = model(MODEL_INPUT_SHAPE=MODEL_INPUT_SHAPE,
                           dp1=dp1,
                           dp2=dp2,
                           dp3=dp3,
                           dp4=dp4,
                           learning_rate=learning_rate,
                           LOSS=LOSS,
                           NUMBER_CLASSES=NUMBER_CLASSES)

    return model_to_train


def train_model(model_to_train, TRAIN_FEATURES, TRAIN_LABELS, TRAIN_STEPS, PREDICT_FEATURES, PREDICT_LABELS, EPOCHS=10):
    print("Traing model...")
    history = model_to_train.fit(
        x=TRAIN_FEATURES,
        y=TRAIN_LABELS,  # batch_size = BATCH_SIZE,
        steps_per_epoch=TRAIN_STEPS,
        shuffle=True,
        # callbacks = Callbacks,
        epochs=EPOCHS,
        # validation_data = train_data_2,
        # validation_data = (TRAIN_FEATURES, TRAIN_LABELS),
        validation_data=(PREDICT_FEATURES, PREDICT_LABELS),
        # validation_steps = TRAIN_STEPS,
        # validation_batch_size= BATCH_SIZE,
        verbose=1,
    )
    print("Done!")

    return model_to_train, history


def save_trained_model_s3bucket_and_log_artifacts(trained_model, history, window, overlap, model_local_path=RNN_TRAIN_DATACLASS.model_local_path, bucket_name=RNN_TRAIN_DATACLASS.bucket_name[0], model_s3_name=RNN_TRAIN_DATACLASS.model_s3_name):

    trained_model.save(model_local_path)
    try:
        # print(print(bucket_name))
        upload_file_to_s3(file_path=model_local_path, bucket_name=bucket_name, object_name=model_s3_name, window=window, overlap=overlap)
    except Exception as e:
        print("Failed to Upload to S3 bucket. Error:", e)
    artifact_path = "./temp/artifacts/1.png"
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.title("Plot of model metrics")
    plt.savefig(artifact_path)
    # plt.show()
    # print("\n\n")


def get_trained_model_confusionM(trained_model, TRAIN_FEATURES, TRAIN_LABELS, PREDICT_FEATURES, PREDICT_LABELS):
    print("----------Confusion matrix on Training samples-----------------")
    features2 = TRAIN_FEATURES
    labs2 = TRAIN_LABELS
    predictions = trained_model.predict(features2)
    pred_1hot = np.argmax(predictions, axis=1)
    pred_true = np.argmax(labs2, axis=1)
    print(confusion_matrix(pred_true, pred_1hot))
    # print(classification_report(pred_true, pred_1hot))
    print("\n\n")

    print("----------Confusion matrix on validation samples-----------------")

    features = PREDICT_FEATURES
    labs = PREDICT_LABELS
    predictions = trained_model.predict(features)
    pred_1hot = np.argmax(predictions, axis=1)
    pred_true = np.argmax(labs, axis=1)
    print(confusion_matrix(pred_true, pred_1hot))
    # print(classification_report(pred_true, pred_1hot))

    # train_task.close()
    


def train_new_model_from_streamlit_ui(train_task, clearml_task_name,sample_window,degree_of_overlap,PERCENT_OF_TRAIN,learning_rate,
                                model_s3_name,LOSS=tf.keras.losses.Huber(),EPOCHS=10):
    (
        TRAIN_FEATURES,
        TRAIN_LABELS,
        TOTAL_TRAIN_DATA,
        PREDICT_FEATURES,
        PREDICT_LABELS,
        TOTAL_VAL_DATA,
        INPUT_FEATURE_SHAPE,
        TRAIN_BATCH_SIZE,
        TRAIN_STEPS,
    ) = initialise_training_variables(
        sample_window=sample_window,
        degree_of_overlap=degree_of_overlap,
        PERCENT_OF_TRAIN=PERCENT_OF_TRAIN,
    )

    model_to_train = initialise_train_model(
        MODEL_INPUT_SHAPE=INPUT_FEATURE_SHAPE,
        dp1=0.3,
        dp2=0.3,
        dp3=0.0,
        dp4=0.0,
        learning_rate=learning_rate,
        LOSS=LOSS,
    )

    trained_model, history = train_model(
        model_to_train,
        TRAIN_FEATURES,
        TRAIN_LABELS,
        TRAIN_STEPS,
        PREDICT_FEATURES,
        PREDICT_LABELS,
        EPOCHS=EPOCHS,
    )

    save_trained_model_s3bucket_and_log_artifacts(
        trained_model,
        history,
        # model_local_path= TRAIN_MODEL.model_local_path,
        # bucket_name=TRAIN_MODEL.bucket_name,
        model_s3_name=str(model_s3_name),
        window=str(sample_window),
        overlap=str(degree_of_overlap),
    )

    get_trained_model_confusionM(
        trained_model,
        TRAIN_FEATURES,
        TRAIN_LABELS,
        PREDICT_FEATURES,
        PREDICT_LABELS,
    )

    st.write("Done training and uploading")

    return train_task.close()










    

    # mlflow.set_tracking_uri("http://127.0.0.1:8080")
    # rnn_experiment = mlflow.set_experiment("rnn_models")
    # run_name = "First Run"

    # metrics = pd.DataFrame(history.history)
    # # print(metrics.describe())

    # metrics = {"loss": metrics['loss'][1], "accuracy": metrics['accuracy'][1], "val_loss": metrics['val_loss'][1], "val_accuracy": metrics['val_accuracy'][1]}


    # #Initiate the MLflow run context
    # with mlflow.start_run(run_name=run_name) as run:
    #     # Log parameters
    #     # mlflow.log_param("epochs", 5)
    #     # mlflow.log_param("optimizer", "adam")
    #     # mlflow.log_params(params_for_mlflow_log)

    #     # Log metrics
    #     mlflow.log_metric("accuracy", history.history['accuracy'][-1])
    #     mlflow.log_metric("loss", history.history['loss'][-1])

    #     # Log artifacts (e.g., saved plots, etc.)
    #     mlflow.log_artifact(artifact_path)

    #     # Save the model in a format that can be loaded later
    #     # model.save("./temp/models/model.h5")
    #     mlflow.log_artifact("/workspaces/Fatique-Detection-From-Physiological-Signals/temp/models/model.h5")
