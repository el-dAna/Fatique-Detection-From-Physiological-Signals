# from dataclasses import dataclass
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.common_functions import (
    train_stack,
    window_sampling,
)
from utils.preprocessingfunctions import (
    get_variables,
)

from utils.rnn_train import RNN_TRAIN_DATACLASS

# from rnn_model import model
# from datetime import datetime
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix #, classification_report



def predict_from_streamlit_data(streamlit_all_data_dict, PATH_TO_SAVED_VARIABLES=RNN_TRAIN_DATACLASS.PATH_TO_SAVED_VARIABLES, NUMBER_CLASSES=RNN_TRAIN_DATACLASS.NUMBER_CLASSES, WINDOW = RNN_TRAIN_DATACLASS.WINDOW, OVERLAP = RNN_TRAIN_DATACLASS.OVERLAP):
    (
        _ ,
        CATEGORIES,
        LABELS_TO_NUMBERS_DICT,
        NUMBERS_TO_LABELS_DICT,
    ) = get_variables(PATH_TO_SAVED_VARIABLES)

    
    WINDOW_SAMPLING_DICT = {
        i: j
        for i, j in enumerate(
            window_sampling(streamlit_all_data_dict, window_size=WINDOW, overlap=OVERLAP)
        )
    }
    TOTAL_GEN_SAMPLES = len(WINDOW_SAMPLING_DICT.keys())
    SAMPLES_PER_SAMPLE = int(TOTAL_GEN_SAMPLES / len(streamlit_all_data_dict.keys()))
    PERCENT_OF_TRAIN = 1 #use all for prediction

    RELAX_PROPORTION = (
        (4) * SAMPLES_PER_SAMPLE
    )  # there are 4 relax sessions for every subject. If SAMPLES_PER_SAMPLE are generated for every subject, then there are RELAX_PROPORTION in total for relax
    OTHERS_PROPORTION = (
        (1) * SAMPLES_PER_SAMPLE
    ) # there is 1 session for any other class for every subject. If SAMPLES_PER_SAMPLE are generated for every subject, then there are OTHERS_PROPORTION in total for other classes

    TRAIN_RELAX_PROPORTION = int(
        PERCENT_OF_TRAIN * RELAX_PROPORTION
    )  # how many of the (number of relax sampled to generate a dataset) are used 
    TRAIN_OTHERS_PROPORTION = int(
        PERCENT_OF_TRAIN * OTHERS_PROPORTION
    )  # how many of the (number of other labels sampled to generate a dataset) are used

    INFERENCE_FEATURES = train_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        # train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=True,
    )

    #TOTAL_TRAIN_DATA = len(TRAIN_FEATURES)

    INFERENCE_LABELS = train_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        # train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=False,
    )

    loaded_model = load_model("./data/models/model.h5")
    predictions = loaded_model.predict(INFERENCE_FEATURES)

    prediction_1hot = np.argmax(predictions, axis=1)
    pred_true = np.argmax(INFERENCE_LABELS, axis=1)
    Confusion_matrix = confusion_matrix(pred_true, prediction_1hot)


    return Confusion_matrix