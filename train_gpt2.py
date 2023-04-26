import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Récupération des noms et des chemins d'accès des fichiers python
def get_files(folder):
    file_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

# Lecture du contenu de chaque fichier et collecte des données
def read_files(file_paths):
    dataset = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
            file_name = os.path.basename(file_path)
            file_dir = os.path.dirname(file_path)
            dataset.append({'file_name': file_name, 'file_dir': file_dir, 'data': data})
    return dataset

# Entraînement du modèle
def train_model(dataset):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    data = [entry['data'] for entry in dataset]
    input_ids = tokenizer.batch_encode_plus(data, pad_to_max_length=True, return_tensors='tf')['input_ids']
    labels = np.concatenate((input_ids[:, 1:], np.zeros((input_ids.shape[0], 1), dtype=int)), axis=-1)
    input_ids = input_ids[:, :-1]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=[loss])
    model.fit(input_ids, labels, batch_size=4, epochs=3)
    return model

# Récupération des noms et des chemins d'accès des fichiers python
folder_path = './data'
file_paths = get_files(folder_path)

# Lecture du contenu de chaque fichier et collecte des données
dataset = read_files(file_paths)

# Entraînement du modèle
model = train_model(dataset)
