# src/utils.py

import os
import pickle
from collections import defaultdict
from PIL import Image

import numpy as np
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTModel
import torch

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_dataset(caption_file, image_folder, vocab_size=5000, max_length=40):
    """
    Load and process the Flickr8k caption dataset.

    Parameters:
    - caption_file (str): Path to the caption .txt file
    - image_folder (str): Folder containing images
    - vocab_size (int): Max number of words to keep in tokenizer
    - max_length (int): Max length of caption sequences

    Returns:
    - tokenizer (Tokenizer): Fitted Keras tokenizer
    - image_caption_mapping (dict): {image_name: [caption1, caption2, ...]}
    """
    image_caption_mapping = defaultdict(list)

    with open(caption_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            img_caption = parts[0]
            caption = parts[1]
            image_name = img_caption.split('#')[0]
            image_caption_mapping[image_name].append(f"startseq {caption.lower()} endseq")

    # Collect all captions for tokenizer
    all_captions = []
    for captions in image_caption_mapping.values():
        all_captions.extend(captions)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="unk")
    tokenizer.fit_on_texts(all_captions)

    return tokenizer, image_caption_mapping


def load_tokenizer(tokenizer_path="tokenizer.pkl"):
    """
    Loads a saved Keras tokenizer from a pickle file.

    Parameters:
    - tokenizer_path (str): Path to tokenizer.pkl

    Returns:
    - tokenizer: Keras Tokenizer object
    """
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def extract_features(image_caption_mapping, image_folder="data/flickr8k/Images"):
    """
    Extract ViT features (CLS token) for all images.

    Parameters:
    - image_caption_mapping (dict): {image_name: [captions]}
    - image_folder (str): Path to images folder

    Returns:
    - features (dict): {image_name: numpy array of CLS token}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    vit_model.eval()

    features = {}
    for image_name in tqdm(image_caption_mapping.keys(), desc="Extracting ViT features"):
        image_path = os.path.join(image_folder, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {image_path}: {e}")
            continue

        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            features[image_name] = cls_token

    return features
