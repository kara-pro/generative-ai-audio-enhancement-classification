import os
import numpy as np
import librosa
import soundfile as sf

def preprocess_audio(relative_path):
    '''
    process audio by normailzing volume and apply pre-emphasis filter
    parameters:
    relative_path -- path to file
    
    returns processed data 
    '''
    (y,sr) = librosa.load(relative_path,mono=False) #standardize sample rate
    #(-32768, 32767)
    #()
    #(y,sr) = librosa.load(out_file,sr=44100,mono=False) 
    max_peak = np.max(np.abs(y)) 
    ratio = 8388607 / max_peak 
    y = y * ratio
    y_filt = librosa.effects.preemphasis(y)
    return y_filt

input = os.getenv('INPUT_PATH')
output = os.getenv("OUTPUT_PATH")
os.chdir(input)
for root, dirs, files in os.walk("."):  
    for file in files:
        folder = os.path.basename(root)
        relative_path = os.path.join(folder, file)
        processed_data = preprocess_audio(relative_path)
        output_folder = os.path.join(output, os.path.basename(folder))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_path = os.path.join(output_folder, file)
        sampling_rate = 16000 #adjust sampling rate on save
        sf.write(output_path, processed_data, sampling_rate, subtype='PCM_24')
