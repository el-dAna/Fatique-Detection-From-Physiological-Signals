import pywt, pickle, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_coefficients(DATADICT, wavelet="morl"):
    # (DATADICT[0])#original shape = (7,300)
    scales = range(1, DATADICT[0].shape[1])
    print(f"Shape of one a DATADICT element: {DATADICT[0].shape}")
    temp1 = []

    for j in range(len(DATADICT.keys())):
        sample1 = np.transpose(DATADICT[j])
        temp2 = []
        for i in range(7):
            coeff, freq = pywt.cwt(sample1[:, i], scales, wavelet)
            temp2.append(coeff)
        temp1.append(temp2)
    Array = np.array(temp1)
    print(f"Shape of big array of coefficents: {Array.shape}")
    return Array


class WaveletDatagenerator(tf.keras.utils.Sequence):
    """
    Datagenerator for the CNN model. This main difference from the physiodatagen is that the sequence windowed features are transformed to images using continouos wavelet transform filter
    check datageneration method for the application

    RETURNS
    X: array -> batched array of CWT features
    Y: array -> corresponding batched one-hot-encoded labels
    """

    number_of_generated_samples = 0

    def __init__(
        self,
        total_subject_num,
        data_dict,
        labels_to_numbers_dict,
        numbers_to_labels_dict,
        input_dimention=(7, 300),
        batch_size=10,
        num_classes=4,
        num_channels=7,
        shuffle=False,
        augment_data=False,
        steps_per_epoch=0.5,
        predict=False,
    ):
        self.total_subject_num = total_subject_num
        self.data_dict = {
            i: j for i, j in enumerate(data_dict)
        }  # this converts the numpy array into a dictionary with their indices as keys
        self.labels_to_numbers_dict = labels_to_numbers_dict
        self.numbers_to_labels_dict = numbers_to_labels_dict
        self.input_dimention = input_dimention
        self.batch_size = batch_size
        self.augment_data = augment_data
        self.steps_per_epoch = steps_per_epoch
        self.predict = predict

        # if assertion thrown, batch_size = 10 is the default size. Check the Train file and pass the right size to the datagenerator
        # assert(self.batch_size < 0.5*self.total_subject_num)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.seed = 0
        print(f"Total samples = {self.total_subject_num}")

        # if overlap_sampling:
        #   self.array_width = 300 #input_dimention[1]
        #   self.percent_overlap = 1-percent_overlap
        #   self.stride = int(percent_overlap*window_size)
        #   assert(self.stride > 0 and self.stride != 1), "Stride too small. Reduce the value of overlap."
        #   self.max_samples = int(((self.array_width - window_size)/self.stride) + 1)
        #   assert(window_size <= self.array_width), f"Window size should be less than total array width. Window_size:{window_size}, array_width:{self.array_width}. Check and make sensitivity=1 so that array has original dimention"
        #   assert(percent_overlap != 1), "Percentage of overlap should be less than 100% or 1"
        #   self.zipped_indices = list( zip((int(percent_overlap*window_size*i) for i in range(self.max_samples)), (int(window_size+percent_overlap*window_size*i) for i in range(self.max_samples))))

        """
    # THIS IS USED FOR VERIFYING THE DATA FROM THE GENERATOR MATCHES
    # THAT OF THE RECORDED
    # self.subject1spo2 = np.hstack([self.data_dict[0], self.data_dict[1], self.data_dict[2], self.data_dict[3]])[0,:]
    # import pickle, os
    # os.rmdir('/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/vars.py')
    # os.mkdir('/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/vars.py') 
    # with open('/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/vars.py', 'wb') as f:
    # # b is added cos the data we are writing may be binary .... in the case of ndarrays
    #   pickle.dump(self.subject1spo2, f)
    """

    def __getitem__(self, index):
        batch_to_load = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # print(f'Batch to load = {batch_to_load}')
        X, y = self.__data_generation(batch_to_load)

        return X, y

    def __len__(self):
        return int((len(self.data_dict.keys())) / self.batch_size)

    def on_epoch_end(self):
        """
        Shuffles the list of indices for random generation
        """
        self.indexes = np.arange(len(self.data_dict.keys()))
        self.seed = np.random.randint(0, 10)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indices):
        """
        The batch indices list contains the indices of the next batch to fetch. [0,1,2,3] fetches the first, sec, third and
        fourth sampples if our batch size is 4
        """
        # this initialises empty values and label array to be filled later
        # X = np.empty((self.batch_size, *self.input_dimention))
        # X = np.empty((self.batch_size, self.data_dict[0].shape[0], self.data_dict[0].shape[1]-1, self.data_dict[0].shape[1] ))
        X = np.empty(
            (
                self.batch_size,
                3,
                self.data_dict[0].shape[1] - 1,
                self.data_dict[0].shape[1],
            )
        )
        y = np.empty((self.batch_size), dtype=int)
        WaveletDatagenerator.number_of_generated_samples += 1

        for i, j in enumerate(batch_indices):
            if (
                self.augment_data
                and (WaveletDatagenerator.number_of_generated_samples % 4) == 0
            ):
                np.random.seed(self.seed)
                np.random.shuffle(self.data_dict[j])
                scales = range(1, self.data_dict[j].shape[1])
                coefs_temp = []
                # for k in range(self.data_dict[j].shape[0]):
                for k in range(3):
                    coefficients, frequencies = pywt.cwt(
                        self.data_dict[j][k, :], scales, "morl"
                    )
                    coefs_temp.append(coefficients)
                X[i,] = np.array(coefs_temp) / 255.0
                # X[i,] = self.data_dict[j]/255.

            else:
                # print('Non augmented')
                # temp_sample = np.random.randint(self.max_samples)
                # X[i,] = self.data_dict[j][:, self.zipped_indices[temp_sample][0]: self.zipped_indices[temp_sample][1] ] # original 7-channel data
                temp = np.transpose(self.data_dict[j])  # shape = (,7)
                # print(f'temp shape: {temp.shape}')
                scales = range(1, temp.shape[0])
                coefs_temp = []

                # for k in range(temp.shape[0]):
                for k in range(3):
                    coefficients, frequencies = pywt.cwt(temp[:, k], scales, "morl")
                    coefs_temp.append(coefficients)
                X[i,] = np.array(coefs_temp) / 255.0

            first_quartile = (
                0.25 * self.total_subject_num
            )  # first quarter of data are 'Relax', indices [0-19], if total number of samples is say 80
            second_quartile = (
                0.5 * self.total_subject_num
            )  # 'second quarter' of data are 'PhysicalStress', indices [20-39]
            third_quartile = (
                0.75 * self.total_subject_num
            )  # 'third quarter' of data are 'EmotionalStress', indices [40-59]

            if j < 720:
                y[i] = self.labels_to_numbers_dict["Relax"]
            elif 720 <= j < 720 + 180:
                y[i] = self.labels_to_numbers_dict["PhysicalStress"]
            elif 720 + 180 <= j < 720 + (180 * 2):
                y[i] = self.labels_to_numbers_dict["EmotionalStress"]
            else:
                y[i] = self.labels_to_numbers_dict["CognitiveStress"]

        """
    remember data_dict is organised this way
    data_dict = {0:[], 1:[], 2:[], -----, 79:[]}
    first 20 samples are relaxex
    next 20 physical
    next 20 emotional
    last 20 cognitive
    We know this from the cration of the dictionary in get_data_dict() function where we stacked them in a specific order
    """
        Y = keras.utils.np_utils.to_categorical(y, num_classes=self.num_classes)
        return X, Y


def plot_coefficients(
    coefficients_array, attributes, G, subject_index=0, specific_subject=True
):
    """
    This plots the coefficients for a specific subject
    """
    if specific_subject:
        for i in range(G.NUMBER_CLASSES):
            sample = coefficients_array[16 * i]
            # sample = coefficients_array[0]
            fig, axs = plt.subplots(1, 7, figsize=(15, 6), facecolor="w", edgecolor="k")
            fig.subplots_adjust(hspace=0.5, wspace=0.001)
            axs = axs.ravel()
            for j in range(7):
                axs[j].matshow(sample[j, :, :])
                axs[j].set_title(attributes[j])
            plt.show()
    else:
        print("mtcheww")
        sample = coefficients_array[subject_index]
        fig, axs = plt.subplots(1, 7, figsize=(15, 6), facecolor="w", edgecolor="k")
        fig.subplots_adjust(hspace=0.5, wspace=0.001)
        axs = axs.ravel()
        for j in range(7):
            axs[j].matshow(sample[j, :, :])
            axs[j].set_title(attributes[j])
        plt.show()


def WL_Model_Labels(num_classes, DATADICT):
    label_temp2 = []
    for i in range(num_classes):
        label_temp1 = np.zeros(num_classes)
        for j in range(int(len(DATADICT.keys()) / num_classes)):
            label_temp1[i] = 1
            label_temp2.append(label_temp1)
    return np.array(label_temp2)


# TRAIN
def simple_train_stack(big_array, train_ratio):
    stop = int(20 * train_ratio)
    relax = big_array[0:stop]
    physical = big_array[20 : 20 + stop]
    emotional = big_array[40 : 40 + stop]
    cognitive = big_array[60 : 60 + stop]
    return np.vstack((relax, physical, emotional, cognitive))


# PREDICTION
def simple_predict_stack(big_array, train_ratio):
    stop = int(20 * train_ratio)
    relax = big_array[stop:20]
    physical = big_array[20 + stop : 40]
    emotional = big_array[40 + stop : 60]
    cognitive = big_array[60 + stop : 80]
    return np.vstack((relax, physical, emotional, cognitive))


def manual_predict(model, coefficents):
    for i in range(coefficents.shape[0]):
        p = model.predict(coefficents[i].reshape(1, 7, 299, 300))[0]
        print(np.where(p == max(p))[0][0] + 1)


def plot_loss_accuracy(
    history,
):
    base = "/content/gdrive/My Drive/PhysioProject1/python-classifier-2020/WL_plots"
    target = os.path.join(base, "WL_acc_loss.png")

    try:
        os.remove(target)
        with open(target, "wb") as File:
            accuracy = history.history["accuracy"]
            loss = history.history["loss"]
            epochs = range(len(loss))

            plt.plot(epochs, accuracy, epochs, loss)
            plt.title = "Accuracy and Loss"
            xlabel = "Epochs"
            plt.legend(["Accuracy", "Loss"])
            plt.savefig(target)
            plt.clf()

            # # Only plot the last 80% of the epochs
            # zoom_split = int(epochs[-1] * 0.2)
            # epochs_zoom = epochs[zoom_split:]
            # accuracy_zoom = accuracy[zoom_split:]
            # loss_zoom = loss[zoom_split:]

            # plt.plot(epochs_zoom, accuracy_zoom, epochs_zoom, loss_zoom)
            # title='Zoomed Accuracy and Loss'
            # xlabel='Epochs'
            # plt.legend('Accuracy', 'Loss')
            # plt.savefig(target)

    except FileNotFoundError:
        with open(target, "wb") as File:
            accuracy = history.history["accuracy"]
            loss = history.history["loss"]
            epochs = range(len(loss))

            plt.plot(epochs, accuracy, epochs, loss)
            plt.title = "Accuracy and Loss"
            xlabel = "Epochs"
            plt.legend(["Accuracy", "Loss"])
            plt.savefig(target)
            plt.clf()


""" FOR PREDICTION"""


def predict(WL_saved_model, test_data):
    # WL_saved_model = tf.keras.models.load_model('/content/gdrive/My Drive/PhysioProject1/python-classifier-2020/model/WL_model')
    Dict = {
        i: j
        for i, j in enumerate(
            ["Relax", "PhysicalStress", "EmotionalStress", "CognitiveStress"]
        )
    }
    for i in range(test_data.shape[0]):
        p = WL_saved_model.predict(test_data[i].reshape(1, 7, 299, 300))[0]
        ind = np.where(p == max(p))[0][0]
        print(f"Predicted Label: {Dict[ind]}")
        if (i + 1) % 4 == 0:
            print("\n")


###################################################################
############### VERIFICATION FUNCTIONS ############################


def plot_data_from_DATADICT(DATADICT, NUM_2_LABELS, wavelet="morl", index=63):
    NUM_2_LABELS = {
        i: j
        for i, j in enumerate(
            ["SpO2", "HeartRate", "AccX", "AccY", "AccZ", "Temp", "EDA"]
        )
    }
    sample = np.transpose(DATADICT[index])  # new shape(300,7),
    # print(sample.shape)
    # assert(sample[:, 0].all() == DATA_DICT[0][0,:].all())
    # N2A = {i:j for i,j in enumerate(['Spo2', 'HeartRate', 'AccX', 'AccY', 'AccZ', 'Temp', 'EDA'])}
    scales = range(1, sample.shape[0])
    fig, axs = plt.subplots(
        1, sample.shape[1], figsize=(15, 6), facecolor="w", edgecolor="k"
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.001)
    axs = axs.ravel()
    for i in range(7):
        coeff, freq = pywt.cwt(sample[:, i], scales, wavelet)
        axs[i].matshow(coeff)
        axs[i].set_title(NUM_2_LABELS[i])


# train_datagen = ImageDataGenerator(#rotation_range = 40,
#                                    #width_shift_range = 0.2,
#                                    #height_shift_range = 0.2,
#                                    #validation_split = 0.2,
#                                    horizontal_flip = False,
#                                    vertical_flip = False,
#                                    data_format = 'channels_first',
#                                    )

# train_generator = train_datagen.flow(x = All_coeffs,
#                                      y = WL_ModelLabels,
#                                      batch_size = 8,
#                                      shuffle = False,
#                                      #save_to_dir = '/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/WL_augImages',
#                                      #save_prefix = 'AugWL',
#                                      #save_format = 'jpeg',
#                                       )
