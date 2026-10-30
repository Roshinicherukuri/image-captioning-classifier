import re
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z ]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return f"startseq {caption} endseq"

def build_tokenizer(captions, vocab_limit=None):
    tokenizer = Tokenizer(num_words=vocab_limit, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    return tokenizer

def create_training_pairs(captions_by_image, tokenizer, max_length):
    """Create teacher-forcing pairs: prefix sequence -> next token."""
    pairs = []
    for image_id, captions in captions_by_image.items():
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                input_seq = pad_sequences([seq[:i]], maxlen=max_length, padding="post")[0]
                pairs.append((image_id, input_seq, seq[i]))
    return pairs

def max_caption_length(captions):
    return max((len(c.split()) for c in captions), default=0)
