from dataclasses import dataclass
import numpy as np
#from typing import Tuple
import tensorflow as tf
import pandas as pd
import mlflow
from mlflow.models import infer_signature


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
    PATH_TO_SAVED_VARIABLES = "./saved_vars.py"

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

    RELAX_PROPORTION = 80 * SAMPLES_PER_SAMPLE
    OTHERS_PROPORTION = 20 * SAMPLES_PER_SAMPLE

    TRAIN_RELAX_PROPORTION = int(PERCENT_OF_TRAIN * RELAX_PROPORTION)
    TRAIN_OTHERS_PROPORTION = int(PERCENT_OF_TRAIN * OTHERS_PROPORTION)


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

    EPOCHS = 80

    dp4 = 0.0
    dp1 = 0.3
    dp2 = 0.3
    dp3 = 0.0
    lr = 0.0002

    """
    dp1 = 0.1
    dp2 = 0.5
    dp3 = 0
    lr = 0.0007
    """

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

    # OPTIMIZER = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.0)
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=lr)  # (1e-05 == 0.00001)
    # OPTIMIZER = tf.keras.optimizers.RMSprop(learning_rate = 0.001)#, clipvalue = 0.01)


# train_data = PhysioDatagenerator(G.TOTAL_TRAIN_DATA, # 140
#                                   G.TRAIN_FEATURES,
#                                   G.LABELS_TO_NUMBERS_DICT,
#                                   G.NUMBERS_TO_LABELS_DICT,
#                                   batch_size = G.TRAIN_BATCH_SIZE,
#                                   shuffle = True,
#                                   input_dimention = G.MODEL_INPUT_SHAPE,
#                                   augment_data = False,
#                                   steps_per_epoch = G.TRAIN_STEPS,
#                                   )

# train_data_2 = PhysioDatagenerator(G.TOTAL_TRAIN_DATA,
#                                   G.TRAIN_FEATURES,
#                                   G.LABELS_TO_NUMBERS_DICT,
#                                   G.NUMBERS_TO_LABELS_DICT,
#                                   batch_size = G.TRAIN_BATCH_SIZE,
#                                   shuffle = False,
#                                   input_dimention = G.MODEL_INPUT_SHAPE,
#                                   augment_data = False,
#                                   steps_per_epoch = G.TRAIN_STEPS,
#                                   )

# val_data = PhysioDatagenerator(G.TOTAL_VAL_DATA,
#                                 G.PREDICT_FEATURES,
#                                 G.LABELS_TO_NUMBERS_DICT,
#                                 G.NUMBERS_TO_LABELS_DICT,
#                                 batch_size = G.VAL_BATCH_SIZE,
#                                 shuffle = False,
#                                 input_dimention = G.MODEL_INPUT_SHAPE,
#                                 augment_data = False,
#                                 )


# d = iter(train_data)
# samp1 = next(d)
# samp2 = next(d)
# samp1 = next(d)
# samp2 = next(d)

# samp1 = next(d)
# samp2 = next(d)
# samp1 = next(d)
# print(samp2[1])
# print(G.TRAIN_FEATURES.shape)
# samp2 = next(d)
# print((samp2[1]))

# """

# Callbacks = [stop_training(), schedule_learningRate]
Callbacks = [stop_training()]

params_for_mlflow_log = {
    "project_dataclass": G,
    "dp1": G.dp1,
    "dp2": G.dp2,
    "dp3": G.dp3,
}

model = model(**params_for_mlflow_log)

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




pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.title("hey")
# plt.savefig('./Plots/Base_model/7.')
plt.show()
print("\n\n")


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


# """


# Set tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params_for_mlflow_log)

    # Log the loss metric
    mlflow.log_metric("metrics", pd.DataFrame(history.history))

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("First train", "rnn")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
