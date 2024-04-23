import os
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import pickle

def extract_features(relative_path):
    '''
    process audio by normailzing volume and apply pre-emphasis filter
    parameters:
    relative_path -- path to file
    
    returns processed data 
    '''
    (y,sr) = librosa.load(relative_path) #standardize sample rate
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return spectrogram, mfccs


csv_data = pd.read_csv(r"..\..\data\UrbanSound8K.csv")
class_dict = dict(zip(csv_data.slice_file_name, csv_data.classID))
os.chdir(r"..\..\data\processed")

for root, dirs, files in os.walk("."): 
    folder = os.path.basename(root)
    spectrograms = []
    mfccs_list = []
    for file in files:
        relative_path = os.path.join(folder, file)
        spect, mfccs = extract_features(relative_path)
        fileclass = class_dict.get(file)

        newdict1 = {'file':file,
                   'fold':folder,
                   'class':fileclass,
                   'spectrogram':spect}
        
        spectrograms.append(newdict1)

        newdict2 = {'file':file,
                   'fold':folder,
                   'class':fileclass,
                   'mfccs':mfccs}
        
        mfccs_list.append(newdict2)

    if "fold" in folder:
        output_folder = os.path.join(r"..\features", os.path.basename(folder))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_path = os.path.join(output_folder, "spectrograms.pickle")
        output_path2 = os.path.join(output_folder, "mfccs.pickle")
        with open(output_path, 'wb') as f:
            pickle.dump(spectrograms, f)
        with open(output_path2, 'wb') as f:
            pickle.dump(mfccs_list, f)

