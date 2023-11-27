import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers

#from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model


def wl_model(project_dataclass):
    """
    This function creates the transfer learning model.

    INPUTS:
    project_dataclass: dataclass -> contains global variables to be used throughout the project

    RETURNS:
    WL_model: keras model -> model for classification
    """

    # setting the image_data_format to suit the dataset
    tf.keras.backend.set_image_data_format(
        "channels_first"
    )  # changes image format to accomodate 7 channelled images with channels first
    # tf.keras.backend.image_data_format() # this output the updated system data format

    # setting parameters for the inception model
    copied_model = InceptionV3(
        input_shape=project_dataclass.MODEL_INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
        classes=4,
    )

    # loading the saved weights
    local_weights_file = "/content/gdrive/MyDrive/PhysioProject1/InceptionV3_weights.h5"
    # copied_model.load_weights(local_weights_file)

    for layer in copied_model.layers:
        layer.trainable = False

    # copied_model.summary() #outputs summary of the inceptionV3 model

    last_layer = copied_model.get_layer(
        "mixed10"
    )  # gets the layer from which the InceptionV3 model output should
    # input to out model. Here last_layer is the 7th. So the data runs from the first to seventh layers only before being parsed into our defined model
    # print(last_layer.output)

    last_output_from_copied_model = (
        last_layer.output
    )  # gets the output of the last layer

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output_from_copied_model)
    # Add a fully connected layer with 1,024* hidden units and ReLU activation
    x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.2
    # x = layers.Dropout(0.2)(x)
    # Add a final softmax layer for classification
    x = layers.Dense(4, activation="softmax")(x)

    # Append the dense network to the base model
    WL_model = Model(copied_model.input, x)

    # for layer in WL_model.layers:
    #   layer.trainable = True

    # WL_model.summary() #prints summary of the overall model

    return WL_model
