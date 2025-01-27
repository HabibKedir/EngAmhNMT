#!/usr/bin/env python
# coding: utf-8

# 1. Import Required Libraries

# In[12]:


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout

# Set file paths
data_dir = "C:/Users/HPC2024/Desktop/data"
amh_path = os.path.join(data_dir, "Amh.txt")  # Amharic file
eng_path = os.path.join(data_dir, "Eng.txt")  # English file

# Load sentences
with open(amh_path, 'r', encoding='utf-8') as f:
    amh_sentences = f.readlines()

with open(eng_path, 'r', encoding='utf-8') as f:
    eng_sentences = f.readlines()

print(f"Loaded {len(amh_sentences)} Amharic sentences and {len(eng_sentences)} English sentences.")

# Tokenization function
def create_tokenizer(sentences):
    tokenizer = Tokenizer(filters='', oov_token="<unk>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer

# Create tokenizers for Amharic and English
amh_tokenizer = create_tokenizer(amh_sentences)
eng_tokenizer = create_tokenizer(eng_sentences)

# Get vocabulary sizes
amh_vocab_size = len(amh_tokenizer.word_index) + 1
eng_vocab_size = len(eng_tokenizer.word_index) + 1
print(f"Amharic vocab size: {amh_vocab_size}, English vocab size: {eng_vocab_size}")

# Preprocess sentences: Convert to sequences and pad
def preprocess_sentences(sentences, tokenizer, max_len=30):
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, maxlen=max_len, padding='post')

max_len = 30
amh_data = preprocess_sentences(amh_sentences, amh_tokenizer, max_len)
eng_data = preprocess_sentences(eng_sentences, eng_tokenizer, max_len)

# Split data into train and validation sets
train_amh, val_amh, train_eng, val_eng = train_test_split(amh_data, eng_data, test_size=0.2, random_state=42)


# 3. Define the Transformer Model

class TransformerNMT(Model):
    def __init__(self, input_vocab_size, target_vocab_size, embed_dim, units):
        super(TransformerNMT, self).__init__()
        # Embedding layers
        self.embedding_input = Embedding(input_vocab_size, embed_dim)
        self.embedding_target = Embedding(target_vocab_size, embed_dim)
        
        # LSTM layers for encoding and decoding
        self.encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        self.decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        
        # Dense layer for final predictions
        self.fc = Dense(target_vocab_size)

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        
        # Encoder
        enc_embedded = self.embedding_input(enc_input)
        enc_output, enc_h, enc_c = self.encoder_lstm(enc_embedded)
        
        # Decoder
        dec_embedded = self.embedding_target(dec_input)
        dec_output, _, _ = self.decoder_lstm(dec_embedded, initial_state=[enc_h, enc_c])
        
        # Final output
        output = self.fc(dec_output)
        return output


# 4. Compile and Train the Model

# Hyperparameters
embed_dim = 256
units = 512
batch_size = 16
epochs = 5

# Initialize model
model = TransformerNMT(amh_vocab_size, eng_vocab_size, embed_dim, units)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    [train_amh, train_eng[:, :-1]], train_eng[:, 1:],  # Shift target sequence for teacher forcing
    validation_data=([val_amh, val_eng[:, :-1]], val_eng[:, 1:]),
    batch_size=batch_size,
    epochs=epochs
)

# 5. Evaluate BLEU Score

# Predict translations
val_predictions = model.predict([val_amh, val_eng[:, :-1]])
val_predictions = np.argmax(val_predictions, axis=-1)

# Compute BLEU score
references = [[sentence] for sentence in val_eng[:, 1:]]  # Shifted target sequences
bleu_score = corpus_bleu(references, val_predictions)
print(f"BLEU Score: {bleu_score:.4f}")
