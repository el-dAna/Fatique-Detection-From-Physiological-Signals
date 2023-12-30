import numpy as np, os, keras
#!pip install scikit-learn
import matplotlib.pyplot as plt
import tensorflow as tf



def adjust_sensitivity(Dict, sensitivity):
    temp_list = []
    for i in range(len(Dict.keys())):
        temp1 = np.array(np.hsplit(Dict[i], sensitivity))
        for j in range(sensitivity):
            temp_list.append(temp1[j])
    return temp_list


def window_sampling(samples_dict, window_size=100, overlap=0.6):
    """
    This function window samples the each feature.

    INPUT:
    samples_dict: dict -> contains the features to be sampled. Each feature is a 7x300 array
    window_size: int -> specifies the windwo size, default is 100 values, this three 7x100 arrays from each 7x300 feature
    overlap: float -> specifies the degree of overlap of windows. 0-no overlap.

    RETURNS:
    temp_samples: list -> a list of all the generated windows from each feature from the sample_dict
    """

    array_width = samples_dict[0].shape[1]  # stores the allowable range of indices
    percent_overlap = (
        1 - overlap
    )  # this is used instead to generate the expected overlap. Using just overlap generates (1-overlap) overlap
    stride = int(percent_overlap * window_size)
    assert stride > 0 and stride != 1, "Stride too small. Reduce the value of overlap."
    max_samples = int(((array_width - window_size) / stride) + 1)
    assert (
        window_size < array_width
    ), "Window size should be less than total array width"
    assert overlap != 1, "Percentage of overlap should be less than 100% or 1"

    print("asfdghagf")


    temp_samples = []  # to keep generated samples

    #temp_dict = {}  # keeps generated indices for debugging
    # keeps a zipped list of generated indices for dubugging----somehow unnecessary
    # zipped_indices = list( zip((int(percent_overlap*window_size*i) for i in range(max_samples)), (int(window_size+percent_overlap*window_size*i) for i in range(max_samples))))
    for j in range(len(samples_dict.keys())):
        for i in range(max_samples):
            start = int(percent_overlap * window_size * i)
            stop = int(window_size + (start))
            # temp_dict[i] = [start, stop]

            assert (
                stop <= array_width
            ), f"Allowabe max_index = {array_width}---Last generated index = {stop}."
            temp = samples_dict[j][:, start:stop]
            temp_samples.append(temp)
    print(
        f"Original subject number = {len(samples_dict.keys())} \nSamples per sample = {max_samples} \nTotal generated = {len(temp_samples)}"
    )
    return temp_samples  # ,temp_dict#, zipped_indices


def train_stack(
    big_dict,
    #train_ratio,
    TRAIN_RELAX_PROPORTION,
    RELAX_PROPORTION,
    OTHERS_PROPORTION,
    TRAIN_OTHERS_PROPORTION,
    sensitivity=False,
    features=True,
):
    """
    This function prepares the traing data from the entire preprocessed features

    INPUTS:
    big_dict: a dict -> contains the entire preprocessed dataset
    train_ratio: float [0-1) -> specifies the percent of train
    sensitivity: int -> specifies the number of samples generated if features are window-sampled. A value of 1 means no window sample, thus 1 generated feature for every 1 feature
    features: bool -> if True, function returns features, if false, functio returns labels

    RETURNS:
    stack: numpy array -> a stack of either train features or corresponding labels(depending on the value of features in the arguments)

    """
    Relax_index, PhysicalStress_index, EmotionalStress_index, CognitiveStress_index = (
        80,
        100,
        120,
        140,
    )
    relax, physical, emotional, cognitive = [], [], [], []
    if features:
        if sensitivity == 1:
            relax_indices = [i for i in range(0, Relax_index)]
            for i in relax_indices:
                relax.append(big_dict[i])

            physical_indices = [
                i for i in range(PhysicalStress_index - 20, PhysicalStress_index)
            ]
            for i in physical_indices:
                physical.append(big_dict[i])

            emotional_indices = [
                i for i in range(EmotionalStress_index - 20, EmotionalStress_index)
            ]
            for i in emotional_indices:
                emotional.append(big_dict[i])

            cognitive_indices = [
                i for i in range(CognitiveStress_index - 20, CognitiveStress_index)
            ]
            for i in cognitive_indices:
                cognitive.append(big_dict[i])

        else:
            for i in range(0, TRAIN_RELAX_PROPORTION):
                relax.append(big_dict[i])

            physical_indices = [
                i
                for i in range(
                    RELAX_PROPORTION, RELAX_PROPORTION + TRAIN_OTHERS_PROPORTION
                )
            ]
            for i in physical_indices:
                physical.append(big_dict[i])

            emotional_indices = [
                i
                for i in range(
                    RELAX_PROPORTION + OTHERS_PROPORTION,
                    RELAX_PROPORTION + OTHERS_PROPORTION + TRAIN_OTHERS_PROPORTION,
                )
            ]
            for i in emotional_indices:
                emotional.append(big_dict[i])

            cognitive_indices = [
                i
                for i in range(
                    RELAX_PROPORTION + OTHERS_PROPORTION * 2,
                    RELAX_PROPORTION + OTHERS_PROPORTION * 2 + TRAIN_OTHERS_PROPORTION,
                )
            ]
            for i in cognitive_indices:
                cognitive.append(big_dict[i])

    elif features == False:
        if sensitivity == 1:
            relax_indices = [i for i in range(0, Relax_index)]
            for i in relax_indices:
                relax.append(np.eye(4)[0])

            physical_indices = [
                i for i in range(PhysicalStress_index - 20, PhysicalStress_index)
            ]
            for i in physical_indices:
                physical.append(np.eye(4)[1])

            emotional_indices = [
                i for i in range(EmotionalStress_index - 20, EmotionalStress_index)
            ]
            for i in emotional_indices:
                emotional.append(np.eye(4)[2])

            cognitive_indices = [
                i for i in range(CognitiveStress_index - 20, CognitiveStress_index)
            ]
            for i in cognitive_indices:
                cognitive.append(np.eye(4)[3])

        else:
            # print(TRAIN_RELAX_PROPORTION)
            relax_indices = [i for i in range(0, TRAIN_RELAX_PROPORTION)]
            for i in relax_indices:
                relax.append(np.eye(4)[0])

            physical_indices = [
                i
                for i in range(
                    RELAX_PROPORTION, RELAX_PROPORTION + TRAIN_OTHERS_PROPORTION
                )
            ]
            for i in physical_indices:
                physical.append(np.eye(4)[1])

            emotional_indices = [
                i
                for i in range(
                    RELAX_PROPORTION + OTHERS_PROPORTION,
                    RELAX_PROPORTION + OTHERS_PROPORTION + TRAIN_OTHERS_PROPORTION,
                )
            ]
            for i in emotional_indices:
                emotional.append(np.eye(4)[2])

            cognitive_indices = [
                i
                for i in range(
                    RELAX_PROPORTION + OTHERS_PROPORTION * 2,
                    RELAX_PROPORTION + OTHERS_PROPORTION * 2 + TRAIN_OTHERS_PROPORTION,
                )
            ]
            for i in cognitive_indices:
                cognitive.append(np.eye(4)[3])

    stack = np.vstack(
        (np.array(relax), np.array(physical), np.array(emotional), np.array(cognitive))
    )
    return stack


# PREDICTION CURRENTLY DEPRICATED. WILL MOST PROBABLY BE DELETED.
def predict_stack(
    big_dict,
    #train_ratio,
    TRAIN_RELAX_PROPORTION,
    RELAX_PROPORTION,
    OTHERS_PROPORTION,
    TRAIN_OTHERS_PROPORTION,
    sensitivity=1,
    features=True,
):
    """
    This function prepares the predict data from the entire preprocessed features

    INPUTS:
    big_dict: a dict -> contains the entire preprocessed dataset
    train_ratio: float [0-1) -> specifies the percent of train. From this the ratio for prediction/validation can be deduced
    sensitivity: int -> specifies the number of samples generated if features are window-sampled. A value of 1 means no window sample, thus 1 generated feature for every 1 feature
    features: bool -> if True, function returns features, if false, functio returns labels

    RETURNS:
    stack: numpy array -> a stack of either train features or corresponding labels(depending on the value of features in the arguments)

    """
    relax, physical, emotional, cognitive = [], [], [], []

    if features:
        if sensitivity == 1:
            pass
            # stop = int(subject_number*train_ratio)
            # #relax_indices = [i for i in range(stop, 20)]
            # for i in range(stop, subject_number):
            #   relax.append(big_dict[i])

            # #physical = big_dict[20+stop :40]
            # #physical_indices = [i for i in range(20+stop, 40)]
            # for i in range(subject_number+stop, subject_number*2):
            #   physical.append(big_dict[i])

            # #emotional = big_dict[40+stop :60]
            # #emotional_indices = [i for i in range(40+stop, 60)]
            # for i in range(subject_number*2+stop, subject_number*3):
            #   emotional.append(big_dict[i])

            # #cognitive = big_dict[60+stop :80]
            # #cognitive_indices = [i for i in range(60+stop, 80)]
            # for i in range(subject_number*3+stop, subject_number*4):
            #   cognitive.append(big_dict[i])

        else:
            # stop = int(subject_number*train_ratio)*sensitivity
            # block = subject_number*sensitivity

            # relax_indices = [i for i in range(stop, block)]
            for i in range(TRAIN_RELAX_PROPORTION, RELAX_PROPORTION):
                relax.append(big_dict[i])

            # physical = big_dict[20+stop :40]
            # physical_indices = [i for i in range(block+stop, block*2)]
            for i in range(
                RELAX_PROPORTION + TRAIN_OTHERS_PROPORTION,
                RELAX_PROPORTION + OTHERS_PROPORTION,
            ):
                physical.append(big_dict[i])

            # emotional = big_dict[40+stop :60]
            # emotional_indices = [i for i in range(block*2+stop, block*3)]
            for i in range(
                RELAX_PROPORTION + OTHERS_PROPORTION + TRAIN_OTHERS_PROPORTION,
                RELAX_PROPORTION + OTHERS_PROPORTION * 2,
            ):
                emotional.append(big_dict[i])

            # cognitive = big_dict[60+stop :80]
            # cognitive_indices = [i for i in range(block*3+stop, block*4)]
            for i in range(
                RELAX_PROPORTION + OTHERS_PROPORTION * 2 + TRAIN_OTHERS_PROPORTION,
                RELAX_PROPORTION + OTHERS_PROPORTION * 3,
            ):
                cognitive.append(big_dict[i])

    elif features == False:
        if sensitivity == 1:
            pass
            # for i in range(stop, block):
            #   relax.append(np.eye(4)[0])

            # for i in range(block+stop, block*2):
            #   physical.append(np.eye(4)[1])

            # for i in range(block*2+stop, block*3):
            #   emotional.append(np.eye(4)[2])

            # for i in range(block*3+stop, block*4):
            #   cognitive.append(np.eye(4)[3])

        else:
            # relax_indices = [i for i in range(stop, block)]
            for i in range(TRAIN_RELAX_PROPORTION, RELAX_PROPORTION):
                relax.append(np.eye(4)[0])

            # physical = big_dict[20+stop :40]
            # physical_indices = [i for i in range(block+stop, block*2)]
            for i in range(
                RELAX_PROPORTION + TRAIN_OTHERS_PROPORTION,
                RELAX_PROPORTION + OTHERS_PROPORTION,
            ):
                physical.append(np.eye(4)[1])

            # emotional = big_dict[40+stop :60]
            # emotional_indices = [i for i in range(block*2+stop, block*3)]
            for i in range(
                RELAX_PROPORTION + OTHERS_PROPORTION + TRAIN_OTHERS_PROPORTION,
                RELAX_PROPORTION + OTHERS_PROPORTION * 2,
            ):
                emotional.append(np.eye(4)[2])

            # cognitive = big_dict[60+stop :80]
            # cognitive_indices = [i for i in range(block*3+stop, block*4)]
            for i in range(
                RELAX_PROPORTION + OTHERS_PROPORTION * 2 + TRAIN_OTHERS_PROPORTION,
                RELAX_PROPORTION + OTHERS_PROPORTION * 3,
            ):
                cognitive.append(np.eye(4)[3])

    stack = np.vstack(
        (np.array(relax), np.array(physical), np.array(emotional), np.array(cognitive))
    )
    return stack


# CALLBACK FUNCTIONS
class stop_training(tf.keras.callbacks.Callback):
    """
    This function specifies when to stop training the model in order to avoid overfitting

    """

    def on_epoch_end(self, logs={}):
        if (logs.get("accuracy") > 0.99) and (logs.get("val_accuracy") > 0.99):
            print(
                "\nReached 94.0% accuracy and over 90% val accuracy -> so cancelling training!"
            )
            self.model.stop_training = True


# another callback function to schedule the learning rate. Used to tune the learning rate for optimisation
schedule_learningRate = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10 ** (epoch / 20)
)





class PhysioDatagenerator(tf.keras.utils.Sequence):
    """
    The data generator for the model.

    RETURNS:
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
        num_channels=1,
        shuffle=False,
        augment_data=False,
        #steps_per_epoch=0.5,
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
        # self.steps_per_epoch = steps_per_epoch

        # if assertion thrown, batch_size = 10 is the default size. Check the Train file and pass the right size to the datagenerator
        # assert(self.batch_size < 0.5*self.total_subject_num)
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.seed = 0
        if self.shuffle:
            print(f"Total train samples = {self.total_subject_num}")
        else:
            print(f"Total validation samples = {self.total_subject_num}")

        # if overlap_sample:
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
        X = np.empty((self.batch_size, *self.input_dimention))
        y = np.empty((self.batch_size), dtype=int)
        PhysioDatagenerator.number_of_generated_samples += 1

        for i, j in enumerate(batch_indices):
            if (
                self.augment_data
                and (PhysioDatagenerator.number_of_generated_samples % 4) == 0
            ):
                np.random.seed(self.seed)
                np.random.shuffle(self.data_dict[j])
                X[i,] = self.data_dict[j]

            else:
                # print('Non augmented')
                # temp_sample = np.random.randint(self.max_samples)
                # X[i,] = self.data_dict[j][:, self.zipped_indices[temp_sample][0]: self.zipped_indices[temp_sample][1] ] # original 7-channel data
                X[i,] = self.data_dict[j]

            first_quartile = (
                80 * self.total_subject_num
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


# DEPRECATED FUNCTIONSSSSSSSSSSSSSS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def plot_learnRate_epoch(epoch_number, history):
    """
    This function plots and saves the learning reate for the trained model

    INPUTS:
    """
    base = "/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/Plots"
    target = os.path.join(base, "learningRate.png")

    try:
        os.remove(target)
        with open(target, "wb") as File:
            lrs = 1e-8 * (10 ** (np.arange(epoch_number) / 20))
            plt.figure(figsize=(10, 6))
            plt.grid(True)
            plt.semilogx(lrs, history.history["loss"])
            plt.tick_params("both", length=10, width=1, which="both")
            # plt.axis([1e-8, 1e-3, 0, 30])
            plt.title("Learning Rate Schedule")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.savefig(target)
            plt.clf()

    except FileNotFoundError:
        with open(target, "wb") as File:
            lrs = 1e-8 * (10 ** (np.arange(epoch_number) / 20))
            plt.figure(figsize=(6, 3))
            plt.grid(True)
            plt.semilogx(lrs, history.history["loss"])
            plt.tick_params("both", length=10, width=1, which="both")
            plt.axis([1e-8, 1e-3, 0, 30])
            plt.title("Learning Rate")
            plt.xlabel("Learning Rate")
            plt.ylabel("Epoch Number")
            plt.savefig(target)
            plt.clf()


def plot_loss_accuracy(
    history,
):
    base = "/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/Plots"
    target = os.path.join(base, "Accuracy_Loss.png")

    try:
        os.remove(target)
        with open(target, "wb") as File:
            accuracy = history.history["accuracy"]
            loss = history.history
            ["loss"]
            epochs = range(len(loss))

            plt.plot(epochs, accuracy, epochs, loss)
            plt.title = "Accuracy and Loss"
            plt.xlabel = "Epochs"
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
