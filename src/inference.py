# src/inference.py

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import ViTFeatureExtractor, ViTModel
import torch

from src.model import create_vit_lstm_model
from src.utils import load_tokenizer

# Paths
TOKENIZER_PATH = "tokenizer.pkl"
MODEL_PATH = "models/vit_lstm_caption_model.h5"
MAX_LENGTH = 40

# Load tokenizer
tokenizer = load_tokenizer(TOKENIZER_PATH)
vocab_size = len(tokenizer.word_index) + 1

# Load trained LSTM model
model = create_vit_lstm_model(vocab_size=vocab_size, max_length=MAX_LENGTH)
model.load_weights(MODEL_PATH)

# Load ViT model and feature extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
vit_model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


def extract_vit_features(image_path):
    """
    Extract CLS token (768-dim) from image using ViT
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = vit_model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    return cls_token  # Shape: (768,)


def generate_caption(image_path):
    """
    Generate a caption for the given image path
    """
    photo = extract_vit_features(image_path)
    input_text = "startseq"

    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH, padding="post")

        y_pred = model.predict([[photo], sequence], verbose=0)
        next_word_index = np.argmax(y_pred)

        next_word = tokenizer.index_word.get(next_word_index, "unk")
        input_text += " " + next_word

        if next_word == "endseq":
            break

    final_caption = input_text.replace("startseq", "").replace("endseq", "").strip()
    return final_caption


if __name__ == "__main__":
    # Example usage
    test_image = "data/flickr8k/Images/1000268201_693b08cb0e.jpg"
    caption = generate_caption(test_image)
    print("Generated Caption:", caption)
