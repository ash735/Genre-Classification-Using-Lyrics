import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

MAX_NUM_WORDS = 3317        # Vocabulary size from the Hindi dataset
MAX_SEQUENCE_LENGTH = 224   # 95th percentile of sequence lengths
EMBEDDING_DIM = 300         # Using cc.hi.300.vec embeddings
VALIDATION_SPLIT = 0.2
EPOCHS = 30
BATCH_SIZE = 128

def load_data(input_csv='/content/drive/MyDrive/songs_happysadcleaned.csv'):
    df = pd.read_csv(input_csv)
    df['cleaned_lyrics'] = df['cleaned_lyrics'].astype(str)
    texts = df['cleaned_lyrics'].tolist()
    labels = df['Genre'].tolist()
    return texts, labels

def prepare_labels(labels):
    genres = sorted(list(set(labels)))
    label_map = {genre: idx for idx, genre in enumerate(genres)}
    y = [label_map[label] for label in labels]
    y = to_categorical(np.asarray(y))
    return y, label_map

def load_hindi_embeddings(embedding_file='/content/drive/MyDrive/cc.hi.300.vec/cc.hi.300.vec'):
    embeddings_index = {}
    with open(embedding_file, encoding='utf8') as f:
        first_line = f.readline().strip().split()
        if len(first_line) == 2:
            # Header found; skip processing this line.
            pass
        else:
            word = first_line[0]
            coefs = np.asarray(first_line[1:], dtype='float32')
            embeddings_index[word] = coefs
        for line in f:
            values = line.rstrip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors in the Hindi embeddings file.")
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, max_num_words=MAX_NUM_WORDS, embedding_dim=EMBEDDING_DIM):
    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def build_lstm_model(embedding_matrix, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS,
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
    model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_lstm_model():
    texts, labels = load_data()
    y, label_map = prepare_labels(labels)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=VALIDATION_SPLIT, random_state=42)
    num_classes = y.shape[1]
    embeddings_index = load_hindi_embeddings()
    embedding_matrix = create_embedding_matrix(word_index, embeddings_index)
    model = build_lstm_model(embedding_matrix, num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=(X_val, y_val), callbacks=[early_stop, reduce_lr])
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"LSTM Model Validation Accuracy: {accuracy:.4f}")
    model.save('/content/drive/MyDrive/lstm_genre_model_hindi.h5')
    with open('/content/drive/MyDrive/tokenizer_hindi.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('/content/drive/MyDrive/label_map_hindi.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    print("Model, tokenizer, and label map saved.")

if __name__ == '__main__':
    train_lstm_model()
