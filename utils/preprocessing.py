import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from preprocessingfunctions import SortSPO2HR, SortAccTempEDA, sanity_check_1, necessary_variables, resize_to_uniform_lengths, sanity_check_2_and_DownSamplingAccTempEDA, get_data_dict, plot_varying_recording_time

if __name__ == "__main__":

  # determines whether to delete previously save file for important variables
  overide_previously_saved_file= sys.argv[1]

  # directory of dataset
  BASE_DIR = '/content/gdrive/MyDrive/PhysioProject1/python-classifier-2020/HealthySubjectsBiosignalsDataSet/'

  # percent of data to be used for training
  PERCENT_OF_TRAIN = 1


  TRAIN_PERCENT = int(PERCENT_OF_TRAIN*len(os.listdir(BASE_DIR)))
  TOTAL_SUBJECT_NUM = len(os.listdir(BASE_DIR)[0:TRAIN_PERCENT])

  SUBJECTS = []
  for i in range(TOTAL_SUBJECT_NUM):
    SUBJECTS.append('Subject'+str(i+1))

  print(f'Subjects {SUBJECTS} used for Training. A total of {len(os.listdir(BASE_DIR))-TOTAL_SUBJECT_NUM} reserved for validation.')

  # calling SortSPO2HR and SortAccTempEDA functions from preprocessing functions
  SPO2HR, SPO2HR_attributes_dict = SortSPO2HR(BASE_DIR, SUBJECTS)
  AccTempEDA, AccTempEDA_attributes_dict = SortAccTempEDA(BASE_DIR, SUBJECTS)

  # plots graphs for the varying lenghts of recorded valeus
  #plot_varying_recording_time(SPO2HR_attributes_dict, AccTempEDA_attributes_dict)

  # Fisrt sanity check to ensure accurate data representayion
  #sanity_check_1(BASE_DIR,SUBJECTS, SPO2HR, AccTempEDA, attribute = 'Temp', Spo2HR = False)

  # gets necessary variables
  SPO2HR_target_size, AccTempEDA_target_size, SPO2HR_attributes, AccTempEDA_attributes, categories, attributes_dict, relax_indices, phy_emo_cog_indices, all_attributes = necessary_variables('takes_nothing__HAHAHA')

  SPO2HR_resized, AccTempEDA_resized = resize_to_uniform_lengths(TOTAL_SUBJECT_NUM, categories, attributes_dict, SPO2HR_target_size, SPO2HR, AccTempEDA_target_size, AccTempEDA)

  AccTempEDA_DownSampled = sanity_check_2_and_DownSamplingAccTempEDA(TOTAL_SUBJECT_NUM, categories, attributes_dict, SPO2HR_target_size, SPO2HR_resized, AccTempEDA_target_size, AccTempEDA_resized, relax_indices, phy_emo_cog_indices )

  ALL_DATA_DICT = get_data_dict(TOTAL_SUBJECT_NUM, categories, attributes_dict, SPO2HR_resized, AccTempEDA_DownSampled)

  LABELS_TO_NUMBERS_DICT = {j:i for i,j in enumerate(categories)}
  NUMBERS_TO_LABELS_DICT = {i:j for i,j in enumerate(categories)}

 

  path_to_saved_vars = './saved_vars.py'

  if os.path.exists(path_to_saved_vars) and overide_previously_saved_file == True:
    print('hey 1')
    print(overide_previously_saved_file)
    os.remove(path_to_saved_vars)
    with open(path_to_saved_vars, 'wb') as dump_site:
      pickle.dump(ALL_DATA_DICT, dump_site)
      pickle.dump(categories, dump_site)
      pickle.dump(LABELS_TO_NUMBERS_DICT, dump_site)
      pickle.dump(NUMBERS_TO_LABELS_DICT, dump_site)
  elif os.path.exists(path_to_saved_vars) == True and overide_previously_saved_file == False:
    print("Hey")
    pass 
  else:
    print('Hey 2')
    with open(path_to_saved_vars, 'wb') as dump_site:
      pickle.dump(ALL_DATA_DICT, dump_site)
      pickle.dump(categories, dump_site)
      pickle.dump(LABELS_TO_NUMBERS_DICT, dump_site)
      pickle.dump(NUMBERS_TO_LABELS_DICT, dump_site)


      
  # PLOTS THE UNSAMPLED VERSION OF SELECTED ATTRIBUTE
  # y1 = np.array(AccTempEDA['Relax']['EDA'][0])
  # x1  = np.arange(len(y1))
  # plt.subplot(1,2,1)
  # plt.plot(x1, y1)
  # plt.title('Unsampled Sub_1 Temp')

  # # PLOTS THE DOWMSAMPLED VERSION OF A SELECTED ATTRIBUTE
  # # The resize_to_uniform function updates the global variables automatically. Will Debug later.
  # y2 = np.array(AccTempEDA['Relax']['EDA'][0])
  # x2  = np.arange(len(y2))
  # plt.subplot(1,2,2)
  # plt.plot(x2, y2) 
  # plt.title('Downsampled Sub_1 Temp') 