import tensorflow as tf
# import numpy as np

from wl_functions import (
    predict_stack,
    get_variables,
    #predict,
    #get_variables,
    #plot_coefficients,
)

from wl_train import G

# PERCENT_OF_TRAIN = 0.8 # Check the WL_train.py file under the dataclass to ensure that its the same ratio
# # it could have been imported but the WL_train will have to also run

# path_2_variables = '/content/gdrive/My Drive/PhysioProject1/train_vars.py'
# BIG_DATA_DICT, NUMBERS_TO_LABELS_DICT, BIG_ALL_COEFFICIENTS, BIG_WL_Model_Labels = get_variables(path_2_variables)

print(G.PERCENT_OF_TRAIN)
PRED_COEFFS = predict_stack(G.BIG_ALL_COEFFICIENTS, G.PERCENT_OF_TRAIN)
PRED_LABELS = predict_stack(G.BIG_WL_Model_Labels, G.PERCENT_OF_TRAIN)

WL_saved_model = tf.keras.models.load_model(
    "/content/gdrive/My Drive/PhysioProject1/python-classifier-2020/model/WL_model"
)
# print('16 inputs were reserved for testing\n First 4: Relax\n Next 4: PhysicalStress\n Next 4: EmotionalStress\n Last 4: CognitiveStress')
# predict(WL_saved_model, PRED_COEFFS)

WL_saved_model.evaluate(PRED_COEFFS, PRED_LABELS)

# plot_coefficients(PRED_COEFFS, G.ATTRIBUTES, G, specific_subject= False )
