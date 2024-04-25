import os
import librosa
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def extract_features(relative_path, relative_path2):
    '''
    process audio by normailzing volume and apply pre-emphasis filter
    parameters:
    relative_path -- path to file
    
    returns processed data 
    '''
    (y,sr) = librosa.load(relative_path) #standardize sample rate
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    (y,sr) = librosa.load(relative_path2) #standardize sample rate
    spectrogram2 = librosa.feature.melspectrogram(y=y, sr=sr)
    #mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return spectrogram, spectrogram2

input = os.getenv("INPUT_PATH")
output = os.getenv("OUTPUT_PATH")

csv_data = pd.read_csv(os.path.join(input, r"UrbanSound8K.csv"))
class_dict = dict(zip(csv_data.slice_file_name, csv_data.classID))
#os.chdir(r"data\processed")

for root, dirs, files in os.walk(os.path.join(input, "processed")): 
    folder = os.path.basename(root)
    spectrograms = []
    label_list = []
    for file in files:
        file_path = os.path.join(folder, file)
        relative_path2 = os.path.join(os.path.join(input, "original"), file_path)
        relative_path = os.path.join(os.path.join(input, "processed"), file_path)

        spect, spect2 = extract_features(relative_path, relative_path2)
        fileclass = class_dict.get(file)

        newdict1 = {'file':file,
                   'fold':folder,
                   'class':fileclass,
                   'original_spectrogram':spect2,
                   'spectrogram':spect}
        
        spectrograms.append(newdict1)
        label_list.append(fileclass)

    if "fold" in folder:
        output_folder = os.path.join(r"data\features", os.path.basename(folder))

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        spect1, spect2 = train_test_split(spectrograms, stratify=label_list, test_size=0.25, random_state=42)

        output_path = os.path.join(output_folder, "spectrograms.pickle")
        output_path2 = os.path.join(output_folder, "spectrograms_test.pickle")
        with open(output_path, 'wb') as f:
            pickle.dump(spect1, f)
        with open(output_path2, 'wb') as f:
            pickle.dump(spect2, f)

