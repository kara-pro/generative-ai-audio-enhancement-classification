import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import mlflow
import mlflow.keras
import pickle
import numpy as np
import os
from models.transformer import create_audio_model
import datetime
from sklearn.model_selection import train_test_split

def load_data():

    data_folds = []
    label_folds = []
    spectrogram_folds = []
    for root, dirs, files in os.walk(r"data\features"):  
        for file in files:
            if "spectrograms.pickle" in file:
                folder = os.path.basename(root)
                folder_path = os.path.join(r"data\features", folder)
                i = int(folder[-1])
                relative_path = os.path.join(folder_path, file)
                with open(relative_path, 'rb') as f:
                    data = pickle.load(f)

                specs = []
                lbls = []
                spec2 = []
                for row in data:
                    specs.append(row['spectrogram'])
                    lbls.append(row['class'])
                    spec2.append(row['original_spectrogram'])

                padded_data = []
                padded_spec = []
                target_shape = (128, 238)

                _, specs = train_test_split(specs,test_size=0.3,random_state=54, shuffle=True, stratify=lbls)
                _, specs2 = train_test_split(spec2, test_size=0.3,random_state=54, shuffle=True, stratify=lbls)
                _, lbls = train_test_split(lbls, test_size=0.3, random_state=54, shuffle=True, stratify=lbls)

                for item in specs:
                    #item = truncate_spectrogram(item)
                    padded_item = tuple(
                        pad_sequences(
                            [array], maxlen=target_shape[1], dtype='float32', padding='post', truncating='post', value=0.0
                        )[0]
                        for array in item
                    )
                    padded_data.append(padded_item)

                for item in specs2:
                    #item = truncate_spectrogram(item)
                    padded_item = tuple(
                        pad_sequences(
                            [array], maxlen=target_shape[1], dtype='float32', padding='post', truncating='post', value=0.0
                        )[0]
                        for array in item
                    )
                    padded_spec.append(padded_item)

                #reshaped_data = (padded_data, (-1, 1))
                #reshaped_data2 = tf.reshape(padded_data, (-1, 1))

                #one_hot_labels = tf.one_hot(lbls, depth=10)
                data_folds.append(padded_data)
                label_folds.append(lbls)
                spectrogram_folds.append(padded_spec)
    return data_folds, label_folds, spectrogram_folds

def load_test():
    data_folds = []
    label_folds = []
    spectrogram_folds = []
    for root, dirs, files in os.walk(r"data\features"):  
        for file in files:
            if "spectrograms_test" in file:
                folder = os.path.basename(root)
                folder_path = os.path.join(r"data\features", folder)
                i = int(folder[-1])
                relative_path = os.path.join(folder_path, file)
                with open(relative_path, 'rb') as f:
                    data = pickle.load(f)

                specs = []
                lbls = []
                spec2 = []
                for row in data:
                    specs.append(row['spectrogram'])
                    lbls.append(row['class'])
                    spec2.append(row['original_spectrogram'])

                padded_data = []
                padded_spec = []
                target_shape = (128, 238)
                _, specs = train_test_split(specs, test_size=0.5, random_state=55, shuffle=True, stratify=lbls)#specs[0:10]
                _, specs2 = train_test_split(spec2, test_size=0.5, random_state=55, shuffle=True, stratify=lbls)#specs[0:10]
                _, lbls = train_test_split(lbls, test_size=0.5, random_state=55, shuffle=True, stratify=lbls)#lbls[0:10]
                for item in specs:
                    #item = truncate_spectrogram(item)
                    padded_item = tuple(
                        pad_sequences(
                            [array], maxlen=target_shape[1], dtype='float32', padding='post', truncating='post', value=0.0
                        )[0]
                        for array in item
                    )
                    padded_data.append(padded_item)

                for item in specs2:
                    #item = truncate_spectrogram(item)
                    padded_item = tuple(
                        pad_sequences(
                            [array], maxlen=target_shape[1], dtype='float32', padding='post', truncating='post', value=0.0
                        )[0]
                        for array in item
                    )
                    padded_spec.append(padded_item)

                #reshaped_data = (padded_data, (-1, 1))
                #reshaped_data2 = tf.reshape(padded_data, (-1, 1))

                #one_hot_labels = tf.one_hot(lbls, depth=10)
                data_folds.extend(padded_data)
                label_folds.extend(lbls)
                spectrogram_folds.extend(padded_spec)
    return data_folds, label_folds, spectrogram_folds


def run_training(epoch_num, model, data_folds, label_folds, spectrogram_folds):

    num_folds = 10
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    chpt_dir = "logs/checkpoint/ckpt.model.keras" 
    callback = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
    checkpoint_callback = ModelCheckpoint(chpt_dir, save_best_only=True)
    reducelr_callback = ReduceLROnPlateau(patience=5, factor=0.0001)
    for e in range(epoch_num):
        print(f"Top Epoch Num {str(e)}")
        for fold in range(num_folds):
            train_data, val_data = [], []
            train_labels, val_labels, train_spectrograms, val_spectrograms = [], [], [], []
           
            for i, (data_fold, label_fold, spectr_fold) in enumerate(zip(data_folds, label_folds, spectrogram_folds)):
                if i == fold:
                    val_data.extend(data_fold)
                    val_labels.extend(label_fold)
                    val_spectrograms.extend(spectr_fold)

                else:
                    train_data.extend(data_fold)
                    train_labels.extend(label_fold)
                    train_spectrograms.extend(spectr_fold)
            # Convert lists to numpy arrays if needed
            train_data = np.array(train_data)
            val_data = np.array(val_data)
            train_labels = np.array(train_labels)
            train_spect = np.array(train_spectrograms)

            val_labels = np.array(val_labels)
            val_spect = np.array(val_spectrograms)

            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, (train_labels, train_spect)))
            # Optionally batch the dataset
            batch_size = 16
            train_dataset = train_dataset.batch(batch_size=batch_size)

            validation_dataset = tf.data.Dataset.from_tensor_slices((val_data, (val_labels, val_spect))).batch(batch_size=batch_size).shuffle(buffer_size=100)
            # Train your model on train_data and train_labels
            model.fit(train_dataset,validation_data=validation_dataset, epochs=2, callbacks=[callback,tensorboard_callback, checkpoint_callback, reducelr_callback]) #define params

    return model


def objective ():
    mlflow.keras.autolog()
    with mlflow.start_run():
        mlflow.set_tag("model", "transformer")
        #mlflow.log_params()
        data_folds, label_folds, spectrogram_folds = load_data()
        data_test, label_test, spectrogram_test = load_test()
        
        model = load_model(r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\logs\checkpoint\ckpt.model.keras")#create_audio_model((128,238), 10)#, num_transformer_blocks=params['num_transformer_blocks'], num_diffusion_steps=params['num_diffusion_steps'])
        
        model = run_training(10, model, data_folds, label_folds, spectrogram_folds)

        test_labels = np.array(label_test)
        test_spect = np.array(spectrogram_test)
        test_data = np.array(data_test)

        train_dataset = tf.data.Dataset.from_tensor_slices((test_data, (test_labels, test_spect))).batch(batch_size=16)
        score = model.evaluate(train_dataset)
        mlflow.log_metric("accuracy", score[1])
        mlflow.log_metric("loss", score[0])

    return {'loss': -score[1], 'status': STATUS_OK, 'model':model}

def main():
    
 
    # search_space= {
    #     #'num_transformer_blocks': hp.randint('num_transformer_blocks', 1, 10),
    #     #'num_diffusion_steps': hp.randint('num_diffusion_steps', 10, 20),
    #     'data_folds': hp.choice('data_folds', [data_folds]),
    #     'label_folds':hp.choice('label_folds', [label_folds]),
    #     'spectrogram_folds':hp.choice('spectrogram_folds', [spectrogram_folds]),
    #     'test_data':hp.choice('test_data', [data_test]),
    #     'label_test':hp.choice('label_test', [label_test]),
    #     'spectrogram_test':hp.choice('spectrogram_test', [spectrogram_test])
    #     }

    mlflow.set_tracking_uri("http://127.0.0.1:5000") #  connects to a tracking URI.
    mlflow.set_experiment("final-project-experiment_training")
    # best_result = fmin(
    #    fn=objective,
    #    space=search_space,
    #    algo=tpe.suggest,
    #    max_evals=1,
    #    trials=Trials()
    # )
    objective()
    
if __name__ == "__main__":
    main()
