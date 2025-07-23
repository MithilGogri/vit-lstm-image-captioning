# src/train.py

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from src.model import create_vit_lstm_model
from src.utils import load_dataset, extract_features

# Paths and hyperparameters
IMAGE_FOLDER = "data/flickr8k/Images"
CAPTION_FILE = "data/flickr8k/Flickr8k.token.txt"
BATCH_SIZE = 32
EPOCHS = 20
MAX_LENGTH = 40
VOCAB_SIZE = 5000

# Step 1: Load tokenizer and image-caption mapping
tokenizer, image_caption_mapping = load_dataset(CAPTION_FILE, IMAGE_FOLDER, VOCAB_SIZE, MAX_LENGTH)

# Step 2: Extract ViT features for all images
print("Extracting ViT features...")
image_features = extract_features(image_caption_mapping)

# Step 3: Prepare training data
X1, X2, y = [], [], []
print("Preparing training sequences...")
for img_name, captions in tqdm(image_caption_mapping.items()):
    feature = image_features[img_name]
    for caption in captions:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=MAX_LENGTH)[0]
            out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

# Step 4: Build and compile the model
model = create_vit_lstm_model(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Step 5: Train the model
model.fit([X1, X2], y, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Step 6: Save the model and tokenizer
model.save("models/vit_lstm_caption_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training complete. Model and tokenizer saved.")
