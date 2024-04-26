from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import numpy as np
import os
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the MLflow model as a PyFunc model
os.getenv('MODEL_PATH')
model = load_model(input)


app = FastAPI()

@app.post('/predict')
def predict(data: str):
    # Assuming data is a dictionary with model input

    #librosa
    (y,sr) = librosa.load(data)
    ls = []
    spectrogram = ls.extend(librosa.feature.melspectrogram(y=y, sr=sr))
    test_spect = np.array(spectrogram)
    train_dataset = tf.data.Dataset.from_tensor_slices(test_spect)

    prediction = model.predict(train_dataset)
    return {"prediction": str(prediction)}