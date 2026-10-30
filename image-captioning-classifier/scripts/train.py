"""
Training scaffold for a Flickr8k-style dataset.

Expected files:
  data/Flickr8k_Dataset/       image files
  data/Flickr8k.token.txt      lines like image.jpg#0<TAB>a caption

This script documents the model-building entry point. For a full experiment,
implement dataset loading, image feature caching, train/validation splitting,
and batched teacher-forcing with your selected dataset.
"""
from pathlib import Path
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from src.data_utils import clean_caption, build_tokenizer, max_caption_length
from src.model import build_caption_model

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "Flickr8k.token.txt"
MODEL_DIR = ROOT / "models"

def main():
    if not ANNOTATIONS.exists():
        raise FileNotFoundError(
            "Dataset annotations not found. See README.md for dataset setup."
        )

    captions = []
    with open(ANNOTATIONS, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                captions.append(clean_caption(parts[1]))

    tokenizer = build_tokenizer(captions)
    max_length = max_caption_length(captions)
    vocab_size = len(tokenizer.word_index) + 1

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    model = build_caption_model(vocab_size, max_length)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.summary()

    print(f"Prepared vocabulary: {vocab_size}")
    print(f"Maximum caption length: {max_length}")
    print("Model architecture initialized.")
    print("Next: implement feature caching and batched training as described in README.md.")

if __name__ == "__main__":
    main()
