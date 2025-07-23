
#  ViT-LSTM Image Captioning

This project demonstrates an image captioning pipeline that combines **Vision Transformers (ViT)** for image feature extraction and **LSTM** for sequence-based caption generation. The model takes an image and generates a descriptive natural language caption.

---

##  Overview

- **Vision Transformer (ViT)** extracts image features (CLS token).
- **LSTM** learns to generate captions conditioned on the image embeddings.
- Trained on the **Flickr8k** dataset with custom tokenizer.

---

##  Project Structure

```
vit-lstm-image-captioning/
├── src/
│   ├── model.py         # ViT-LSTM model architecture
│   ├── utils.py         # Tokenizer, dataset processing, feature extraction
│   ├── train.py         # Model training loop
│   └── inference.py     # Caption generation from image
├── results/
│   └── sample_outputs.txt  # Real outputs: actual + generated captions
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

---

##  Sample Outputs

The file [`results/sample_outputs.txt`](./results/sample_outputs.txt) contains examples of:
-  Image filenames
-  Ground-truth captions
-  Model-generated captions

These outputs are extracted from actual results in the project report.

---

##  How to Use

> **Training and inference were run in Google Colab using the Kaggle Flickr8k dataset.**

1. Download the dataset from [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Run `train.py` in Colab to:
   - Extract ViT features
   - Train the LSTM captioning model
3. Save:
   - `vit_lstm_caption_model.h5`
   - `tokenizer.pkl`
4. Use `inference.py` to generate new captions

---

##  Notes

- The trained model and tokenizer are **not included** due to size.
- However, the codebase is modular and ready-to-train using Colab.
- Real output examples are shown in `sample_outputs.txt`.

---

##  Dataset

- **Flickr8k Dataset**: 8,000 images + 5 captions per image
- Download: [Kaggle - adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

---

##  Author

**Mithil Gogri**  
🔗 [LinkedIn](https://www.linkedin.com/in/mithil-gogri-2615a9288/) | 📧 mithilgogri@gmail.com
