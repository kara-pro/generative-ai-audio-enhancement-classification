import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling2D
import pickle
import numpy as np
from sklearn.metrics import f1_score
import os

class ExtractPatches(Layer):
    def __init__(self, **kwargs):
        super(ExtractPatches, self).__init__(**kwargs)

    def call(self, inputs):
        shape = tf.shape(inputs)
        reshaped_data = tf.reshape(inputs, (-1, shape[1], shape[2], 1))
        patches = tf.image.extract_patches(reshaped_data, sizes=[1, 16, 16, 1], strides=[1, 8, 8, 1], rates=[1,1,1,1], padding='SAME')
        return patches

def embed(data):
   patches = ExtractPatches()(data)
   dense = Dense(128, activation='relu')(patches)
   return dense

# Build the Transformer Block

def transformer_block(x):
    # MultiHead Attention (add only one MultiHeadAttention with 2 heads, add droput and LayerNormalization)
    attention = MultiHeadAttention(2, key_dim = 128) (x, x)
    attention = Dropout(.2)(attention)
    out1 = LayerNormalization(epsilon=1e-6) (x + attention)

    # Feed Forward Network (add fully connected layers)
    ffn_output = Dense(128, activation = 'relu')(out1)
    ffn_output = Dropout(.2)(ffn_output)

    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Assemble full model

def build_model():
    input = Input(shape = (128,238))
    embedding_layer = embed(input)

    x = transformer_block(embedding_layer)
    x = GlobalAveragePooling2D()(x)
    x =  Dropout(0.1)(x)# add droput layer
    outputs = Dense(10, activation='sigmoid')(x)#dende layer

    model = Model(input, outputs) # define inputs and outputs
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy', 'f1_score']) # define params
    return model

def generate_folds():
    os.chdir(r"..\..\data\features")
    data_folds = []
    label_folds = []
    for root, dirs, files in os.walk("."):  
        for file in files:
            if "spectrograms" in file:
                folder = os.path.basename(root)
                i = int(folder[-1])
                relative_path = os.path.join(folder, file)
                with open(relative_path, 'rb') as f:
                    data = pickle.load(f)

                specs = []
                lbls = []
                for row in data:
                    specs.append(row['spectrogram'])
                    lbls.append(row['class'])

                padded_data = []
                target_shape = (128, 238)
                specs = specs
                lbls = lbls
                for item in specs:
                    padded_item = tuple(
                        pad_sequences(
                            [array], maxlen=target_shape[1], dtype='float32', padding='post', truncating='post', value=0.0
                        )[0]
                        for array in item
                    )
                    padded_data.append(padded_item)

                one_hot_labels = tf.one_hot(lbls, depth=10)
                data_folds.append(padded_data)
                label_folds.append(one_hot_labels)

    return data_folds, label_folds

def train_model(model, data_folds, label_folds):
    num_folds = 10

    fold_losses = []
    fold_accuracies = []
    fold_f1_scores = []
    for fold in range(num_folds):
        train_data, val_data = [], []
        train_labels, val_labels = [], []
        for i, (data_fold, label_fold) in enumerate(zip(data_folds, label_folds)):
            if i == fold:
                val_data.extend(data_fold)
                val_labels.extend(label_fold)
            else:
                train_data.extend(data_fold)
                train_labels.extend(label_fold)

        # Convert lists to numpy arrays if needed
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)

        

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        # Optionally batch the dataset
        batch_size = 32
        train_dataset = train_dataset.batch(batch_size=batch_size)

        validation_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size=batch_size).shuffle(buffer_size=100)


        # Train your model on train_data and train_labels
        model.fit(train_dataset, epochs=10) #define params
        # Evaluate your model on val_data and val_labels
        results = model.evaluate(validation_dataset)
        fold_loss = results[0]
        fold_accuracy = results[1]

        fold_losses.append(fold_loss)
        fold_accuracies.append(fold_accuracy)
        fold_predictions = model.predict(val_data)
        val_labels_indices = np.argmax(val_labels, axis=1)
        fold_f1 = f1_score(val_labels_indices, fold_predictions.argmax(axis=1), average='macro')
        fold_f1_scores.append(fold_f1)
        # Calculate performance metric (e.g., accuracy) for this fold

    # Calculate average performance metric across all folds
    #average_accuracy = ...
    avg_loss = sum(fold_losses) / len(fold_losses)
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    avg_f1_score = sum(fold_f1_scores) / len(fold_f1_scores)

    print("Average Loss:", avg_loss)
    print("Average Accuracy:", avg_accuracy)
    print("Average F1 Score:", avg_f1_score)
    return model


def main():
    model = build_model()
    data_folds, label_folds = generate_folds()
    model = train_model(model, data_folds, label_folds)
    os.chdir(r"..\..\src\models")
    model.save('transformer.keras')





    
    

