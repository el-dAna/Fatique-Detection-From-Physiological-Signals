from dataclasses import dataclass
import numpy as np
#from typing import Tuple
import tensorflow as tf
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow import MlflowClient
from pprint import pprint


from common_functions import (
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
from preprocessingfunctions import (
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
from rnn_model import model
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix #, classification_report


tf.keras.backend.clear_session()  # clears internal variables so we start all initiations and assignments afresh


@dataclass
class G:
    BASE_DIR = "./HealthySubjectsBiosignalsDataSet/"
    PATH_TO_SAVED_VARIABLES = "./utils/saved_vars.py"

    (
        WHOLE_DICT,
        CATEGORIES,
        LABELS_TO_NUMBERS_DICT,
        NUMBERS_TO_LABELS_DICT,
    ) = get_variables(PATH_TO_SAVED_VARIABLES)
    # SAVED_CWT_DICT = {i:j/255. for i,j in enumerate(SAVED_CWT_DICT['features'])}
    NUMBER_CLASSES = 4

    WINDOW = 60
    OVERLAP = 0.5
    WINDOW_SAMPLING_DICT = {
        i: j
        for i, j in enumerate(
            window_sampling(WHOLE_DICT, window_size=WINDOW, overlap=OVERLAP)
        )
    }
    TOTAL_GEN_SAMPLES = len(WINDOW_SAMPLING_DICT.keys())
    SAMPLES_PER_SAMPLE = int(TOTAL_GEN_SAMPLES / len(WHOLE_DICT.keys()))
    PERCENT_OF_TRAIN = 0.80

    RELAX_PROPORTION = 80 * SAMPLES_PER_SAMPLE #there are originally 80 features labeled as relax
    OTHERS_PROPORTION = 20 * SAMPLES_PER_SAMPLE #there are originally 20 features labeled as others(physicalstress, emotionalstress and cognituvestress)

    TRAIN_RELAX_PROPORTION = int(PERCENT_OF_TRAIN * RELAX_PROPORTION) #how many of the (number of relax sampled to generate a dataset) are used for training 
    TRAIN_OTHERS_PROPORTION = int(PERCENT_OF_TRAIN * OTHERS_PROPORTION) #how many of the (number of other labels sampled to generate a dataset) are used for training 


    TRAIN_FEATURES = train_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        #train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=True,
    )

    TOTAL_TRAIN_DATA = len(TRAIN_FEATURES)
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

    MODEL_INPUT_SHAPE = TRAIN_FEATURES[0].shape

    EPOCHS = 10

    dp4 = 0.0
    dp1 = 0.3
    dp2 = 0.3
    dp3 = 0.0
    learning_rate = 0.0002

    TRAIN_BATCH_SIZE = int(TOTAL_TRAIN_DATA / 8)  # /8
    assert (
        TOTAL_TRAIN_DATA % TRAIN_BATCH_SIZE == 0
    ), "Ensure that the batch size is perfectly divisible by total_train_data"

    # VAL_BATCH_SIZE = int(TOTAL_VAL_DATA) # /4
    # assert(TOTAL_VAL_DATA % VAL_BATCH_SIZE == 0), "Ensure teh val_batch_size is perfectly divisible by the total_val_data"

    # NUMBER_CLASSES = 4
    TRAIN_STEPS = int(TOTAL_TRAIN_DATA // TRAIN_BATCH_SIZE)
    # VAL_STEPS = int(TOTAL_VAL_DATA // VAL_BATCH_SIZE)

    LOSS = tf.keras.losses.Huber()
    # LOSS = tf.keras.losses.CategoricalCrossentropy()


# Callbacks = [stop_training(), schedule_learningRate]
Callbacks = [stop_training()]

params_for_mlflow_log = {
    "dp1": G.dp1,
    "dp2": G.dp2,
    "dp3": G.dp3,
    "learning_rate":G.learning_rate
}

model = model(G, **params_for_mlflow_log)

print("Traing model...")
history = model.fit(
    x=G.TRAIN_FEATURES,
    y=G.TRAIN_LABELS,  # batch_size = G.BATCH_SIZE,
    steps_per_epoch=G.TRAIN_STEPS,
    shuffle=True,
    # callbacks = Callbacks,
    epochs=G.EPOCHS,
    # validation_data = train_data_2,
    # validation_data = (G.TRAIN_FEATURES, G.TRAIN_LABELS),
    validation_data=(G.PREDICT_FEATURES, G.PREDICT_LABELS),
    # validation_steps = G.TRAIN_STEPS,
    # validation_batch_size= G.BATCH_SIZE,
    verbose=1,
)

print("Done!")



artifact_path = "/workspaces/Fatique-Detection-From-Physiological-Signals/data/artifacts/1.png"
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title("hey")
plt.savefig(artifact_path)
# plt.show()
# print("\n\n")


print("----------Confusion matrix on Training samples-----------------")
features2 = G.TRAIN_FEATURES
labs2 = G.TRAIN_LABELS
predictions = model.predict(features2)
pred_1hot = np.argmax(predictions, axis=1)
pred_true = np.argmax(labs2, axis=1)
print(confusion_matrix(pred_true, pred_1hot))
# print(classification_report(pred_true, pred_1hot))
print("\n\n")

print("----------Confusion matrix on validation samples-----------------")

features = G.PREDICT_FEATURES
labs = G.PREDICT_LABELS
predictions = model.predict(features)
pred_1hot = np.argmax(predictions, axis=1)
pred_true = np.argmax(labs, axis=1)
print(confusion_matrix(pred_true, pred_1hot))
# print(classification_report(pred_true, pred_1hot))


mlflow.set_tracking_uri("http://127.0.0.1:8080")
rnn_experiment = mlflow.set_experiment("rnn_models")
run_name = "First Run"

metrics = pd.DataFrame(history.history)
# print(metrics.describe())

metrics = {"loss": metrics['loss'][1], "accuracy": metrics['accuracy'][1], "val_loss": metrics['val_loss'][1], "val_accuracy": metrics['val_accuracy'][1]}


#Initiate the MLflow run context
with mlflow.start_run(run_name=run_name) as run:
    # Log parameters
    # mlflow.log_param("epochs", 5)
    # mlflow.log_param("optimizer", "adam")
    # mlflow.log_params(params_for_mlflow_log)

    # Log metrics
    mlflow.log_metric("accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("loss", history.history['loss'][-1])

    # Log artifacts (e.g., saved plots, etc.)
    mlflow.log_artifact(artifact_path)

    # Save the model in a format that can be loaded later
    model.save("/workspaces/Fatique-Detection-From-Physiological-Signals/data/models/model.h5")
    mlflow.log_artifact("/workspaces/Fatique-Detection-From-Physiological-Signals/data/models/model.h5")