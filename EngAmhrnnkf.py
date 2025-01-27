#!/usr/bin/env python
# coding: utf-8

# Python implementation for an English to Amharic Neural Machine Translation (NMT) system using rnn in kfold validation split

# In[1]:


get_ipython().system('pip install tensorflow transformers')


# Set directory paths
data_dir = "C:/Users/HPC2024/Desktop/data"
eng_path = os.path.join(data_dir, "Eng.txt")
amh_path = os.path.join(data_dir, "Amh.txt")

# Load data
with open(eng_path, 'r', encoding='utf-8') as f:
    eng_sentences = f.readlines()

with open(amh_path, 'r', encoding='utf-8') as f:
    amh_sentences = f.readlines()

print(f" {len(eng_sentences)} English sentences and Loaded {len(amh_sentences)} Amharic sentences.")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Tokenization
def create_tokenizer(sentences):
    tokenizer = Tokenizer(filters='', oov_token="<unk>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer

eng_tokenizer = create_tokenizer(eng_sentences)
amh_tokenizer = create_tokenizer(amh_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
amh_vocab_size = len(amh_tokenizer.word_index) + 1


# Preprocess sentences
def preprocess_sentences(sentences, tokenizer, max_length=30):
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

max_length = 30

eng_data = preprocess_sentences(eng_sentences, eng_tokenizer, max_length)
amh_data = preprocess_sentences(amh_sentences, amh_tokenizer, max_length)

print(f" English data shape: {eng_data.shape},Amharic data shape: {amh_data.shape}")

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense


# Define the Seq2Seq Model
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embedding_dim, units):
        super(Seq2SeqModel, self).__init__()
        self.encoder_embedding = Embedding(encoder_vocab_size, embedding_dim)
        self.encoder_lstm = LSTM(units, return_state=True)

        self.decoder_embedding = Embedding(decoder_vocab_size, embedding_dim)
        self.decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        self.fc = Dense(decoder_vocab_size)

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs

        # Encoder
        enc_embedded = self.encoder_embedding(enc_input)
        _, enc_h, enc_c = self.encoder_lstm(enc_embedded)

        # Decoder
        dec_embedded = self.decoder_embedding(dec_input)
        dec_output, _, _ = self.decoder_lstm(dec_embedded, initial_state=[enc_h, enc_c])
        output = self.fc(dec_output)

        return output

get_ipython().system('pip install nltk scikit-learn')
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from sklearn.model_selection import KFold
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

# BLEU metric (unchanged)
def calculate_bleu(predictions, references):
    references = [[ref] for ref in references]
    return corpus_bleu(references, predictions)

# K-Fold Split (moved here to be before its usage)
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

embedding_dim = 256
units = 512
batch_size = 16
epochs = 5
bleu_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(eng_data)):
    print(f"Training on Fold {fold + 1}/{num_folds}...")

    # Split data
    train_eng, val_eng = eng_data[train_idx], eng_data[val_idx]
    train_amh, val_amh = amh_data[train_idx], amh_data[val_idx]

    # Define the model
    model = Seq2SeqModel(eng_vocab_size, amh_vocab_size, embedding_dim, units)

    # Compile the model
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss_object,
                  metrics=['accuracy'])

    # Train the model
    model.fit(
        [train_eng, train_amh[:, :-1]], train_amh[:, 1:],
        validation_data=([val_eng, val_amh[:, :-1]], val_amh[:, 1:]),
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluate on validation set
    val_predictions = model.predict([val_eng, val_amh[:, :-1]])
    val_predictions = tf.argmax(val_predictions, axis=-1).numpy()

    bleu = calculate_bleu(val_predictions, val_amh[:, 1:])
    bleu_scores.append(bleu)
    print(f"BLEU score for Fold {fold + 1}: {bleu:.4f}")

# Average BLEU score
print(f"Average BLEU score across folds: {np.mean(bleu_scores):.4f}")
