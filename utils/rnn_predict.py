from dataclasses import dataclass


from common_functions import (
    train_stack,
    window_sampling,
)
from preprocessingfunctions import (
    get_variables,
)

# from rnn_model import model
# from datetime import datetime
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix #, classification_report


@dataclass
class INFERENCE_DATACLASS:
    BASE_DIR = "./HealthySubjectsBiosignalsDataSet/"
    PATH_TO_SAVED_VARIABLES = "./utils/saved_vars.py"

    (
        WHOLE_DICT,
        CATEGORIES,
        LABELS_TO_NUMBERS_DICT,
        NUMBERS_TO_LABELS_DICT,
    ) = get_variables(PATH_TO_SAVED_VARIABLES)

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

    RELAX_PROPORTION = (
        80 * SAMPLES_PER_SAMPLE
    )  # there are originally 80 features labeled as relax
    OTHERS_PROPORTION = (
        20 * SAMPLES_PER_SAMPLE
    )  # there are originally 20 features labeled as others(physicalstress, emotionalstress and cognituvestress)

    TRAIN_RELAX_PROPORTION = int(
        PERCENT_OF_TRAIN * RELAX_PROPORTION
    )  # how many of the (number of relax sampled to generate a dataset) are used for training
    TRAIN_OTHERS_PROPORTION = int(
        PERCENT_OF_TRAIN * OTHERS_PROPORTION
    )  # how many of the (number of other labels sampled to generate a dataset) are used for training

    TRAIN_FEATURES = train_stack(
        big_dict=WINDOW_SAMPLING_DICT,
        # train_ratio=PERCENT_OF_TRAIN,
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
        # train_ratio=PERCENT_OF_TRAIN,
        sensitivity=SAMPLES_PER_SAMPLE,
        TRAIN_RELAX_PROPORTION=TRAIN_RELAX_PROPORTION,
        RELAX_PROPORTION=RELAX_PROPORTION,
        OTHERS_PROPORTION=OTHERS_PROPORTION,
        TRAIN_OTHERS_PROPORTION=TRAIN_OTHERS_PROPORTION,
        features=False,
    )
