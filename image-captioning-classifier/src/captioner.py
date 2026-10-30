from pathlib import Path
import pickle

import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "caption_model.keras"
TOKENIZER_PATH = ROOT / "models" / "tokenizer.pkl"
MAX_LENGTH = 34
IMAGE_SIZE = (299, 299)

class ImageCaptioner:
    """CNN feature extractor + trained sequence decoder."""

    def __init__(self):
        if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                "Trained files are missing. Follow the training instructions in README.md "
                "to create models/caption_model.keras and models/tokenizer.pkl."
            )
        from tensorflow.keras.models import load_model
        self.decoder = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as file:
            self.tokenizer = pickle.load(file)
        self.encoder = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

    def _features(self, image):
        image = image.resize(IMAGE_SIZE)
        array = img_to_array(image)
        array = np.expand_dims(array, axis=0)
        array = preprocess_input(array)
        return self.encoder.predict(array, verbose=0)

    def predict(self, image, max_length=MAX_LENGTH):
        features = self._features(image)
        sequence = [self.tokenizer.word_index["startseq"]]
        for _ in range(max_length):
            padded = pad_sequences([sequence], maxlen=max_length, padding="post")
            prediction = self.decoder.predict([features, padded], verbose=0)
            next_id = int(np.argmax(prediction[0]))
            word = self.tokenizer.index_word.get(next_id)
            if word is None or word == "endseq":
                break
            sequence.append(next_id)
        words = [self.tokenizer.index_word.get(i, "") for i in sequence[1:]]
        return " ".join(word for word in words if word).strip().capitalize()
