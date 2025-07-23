# src/model.py

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model

def create_vit_lstm_model(vocab_size, max_length):
    """
    Creates the ViT-LSTM image captioning model.

    Parameters:
    - vocab_size: Total number of words in the vocabulary.
    - max_length: Maximum length of caption sequences.

    Returns:
    - model: A compiled Keras model ready for training.
    """

    # Image feature input (output from Vision Transformer)
    image_input = Input(shape=(768,), name="image_features_input")
    x1 = Dropout(0.5)(image_input)
    x2 = Dense(256, activation='relu')(x1)

    # Caption input (sequence of word indices)
    caption_input = Input(shape=(max_length,), name="caption_input")
    y1 = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
    y2 = Dropout(0.5)(y1)
    y3 = LSTM(256)(y2)

    # Combine image and text features
    combined = concatenate([x2, y3])
    z1 = Dense(256, activation='relu')(combined)
    output = Dense(vocab_size, activation='softmax')(z1)

    # Define the model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model
