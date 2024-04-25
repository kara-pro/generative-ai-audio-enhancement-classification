import os
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import soundfile as sf

def augement_audio(sample_file, sampling_rate):
    '''
    add noise to the audio sample with audioentations

    parameters:
    sample_file -- np array of audio file (loaded in with librosa)
    sampling_rate -- sampling rate given by librosa upon load
    
    returns augmented data
    '''
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.00075, max_amplitude=0.02, p=0.5),
        TimeStretch(min_rate=0.5, max_rate=1.5, p=0.5),
        PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
        Shift(p=0.5),
    ])
    return augment(samples=sample_file, sample_rate=sampling_rate)

input = os.getenv('INPUT_PATH')
output = os.getenv("OUTPUT_PATH")

os.chdir(os.path.join(input))
for root, dirs, files in os.walk("."):  
    for file in files:
        folder = os.path.basename(root)
        relative_path = os.path.join(folder, file)
        sample_file, sampling_rate = librosa.load(relative_path)
        augmented_data = augement_audio(sample_file, sampling_rate)
        output_folder = os.path.join(output, os.path.basename(folder))
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_path = os.path.join(output_folder, file)
        sf.write(output_path, augmented_data, sampling_rate, subtype='PCM_24')  
      
    
 

