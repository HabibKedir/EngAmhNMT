#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# Load Data
data_dir = "C:/Users/HPC2024/Desktop/data"
eng_path = os.path.join(data_dir, "Eng.txt")
amh_path = os.path.join(data_dir, "Amh.txt")

# Read sentences
with open(eng_path, 'r', encoding='utf-8') as f:
    eng_sentences = f.readlines()

with open(amh_path, 'r', encoding='utf-8') as f:
    amh_sentences = f.readlines()

print(f"Loaded {len(eng_sentences)} English sentences and {len(amh_sentences)} Amharic sentences.")

# Tokenization
def create_tokenizer(sentences):
    tokenizer = Tokenizer(filters='', oov_token="<unk>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer

eng_tokenizer = create_tokenizer(eng_sentences)
amh_tokenizer = create_tokenizer(amh_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
amh_vocab_size = len(amh_tokenizer.word_index) + 1

# Preprocess Sentences
def preprocess_sentences(sentences, tokenizer, max_len=30):
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, maxlen=max_len, padding='post')

max_len = 30
eng_data = preprocess_sentences(eng_sentences, eng_tokenizer, max_len)
amh_data = preprocess_sentences(amh_sentences, amh_tokenizer, max_len)

print(f"English data shape: {eng_data.shape}, Amharic data shape: {amh_data.shape}")

# Train-Test Split
train_eng, val_eng, train_amh, val_amh = train_test_split(eng_data, amh_data, test_size=0.2, random_state=42)

# Transformer Encoder-Decoder Model
class TransformerNMT(Model):
    def __init__(self, input_vocab_size, target_vocab_size, embed_dim, units):
        super(TransformerNMT, self).__init__()
        self.embedding_input = Embedding(input_vocab_size, embed_dim)
        self.embedding_target = Embedding(target_vocab_size, embed_dim)

        self.encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        self.decoder_lstm = LSTM(units, return_sequences=True, return_state=True)

        self.fc = Dense(target_vocab_size)

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs

        # Encoder
        enc_embedded = self.embedding_input(enc_input)
        enc_output, enc_h, enc_c = self.encoder_lstm(enc_embedded)

        # Decoder
        dec_embedded = self.embedding_target(dec_input)
        dec_output, _, _ = self.decoder_lstm(dec_embedded, initial_state=[enc_h, enc_c])

        # Dense layer
        output = self.fc(dec_output)

        return output

# Initialize Model
embed_dim = 256
units = 512
batch_size = 16
epochs = 5

model = TransformerNMT(eng_vocab_size, amh_vocab_size, embed_dim, units)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
history = model.fit(
    [train_eng, train_amh[:, :-1]], train_amh[:, 1:],
    validation_data=([val_eng, val_amh[:, :-1]], val_amh[:, 1:]),
    batch_size=batch_size,
    epochs=epochs
)

# Evaluate BLEU Score
val_predictions = model.predict([val_eng, val_amh[:, :-1]])
val_predictions = np.argmax(val_predictions, axis=-1)

references = [[sentence] for sentence in val_amh[:, 1:]]
bleu_score = corpus_bleu(references, val_predictions)
print(f"BLEU Score: {bleu_score:.4f}")

# Plot Training History
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

