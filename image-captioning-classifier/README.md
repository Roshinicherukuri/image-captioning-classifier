# Image Captioning Classifier

An image captioning project combining computer vision and natural language processing. A CNN extracts visual features from an image, and an LSTM-based decoder generates a word sequence describing its contents.

## Features

- CNN image feature extraction using pretrained InceptionV3
- LSTM sequence decoder for natural-language caption generation
- Caption cleaning and tokenizer utilities
- Streamlit interface for image upload and caption display
- Training model architecture scaffold
- Dataset setup and experiment guidance

## Tech stack

Python · TensorFlow/Keras · CNN · LSTM · Deep Learning · NLP · Streamlit

## Architecture

```text
Input image
    |
    v
InceptionV3 CNN (pretrained feature extractor)
    |
    v
Image feature vector -------------------+
                                         |
Caption prefix -> Embedding -> LSTM -----+--> Dense + Softmax
                                                    |
                                                    v
                                            Next word prediction
                                                    |
                                                    v
                                             Generated caption
```

The decoder is trained with teacher forcing: given image features and the previous caption tokens, it learns to predict the next token.

## Project structure

```text
image-captioning-classifier/
├── app.py
├── src/
│   ├── captioner.py
│   ├── data_utils.py
│   └── model.py
├── scripts/
│   └── train.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or 3.11 is recommended; choose a TensorFlow version compatible with your operating system and Python version.

```bash
git clone https://github.com/<your-username>/image-captioning-classifier.git
cd image-captioning-classifier

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

This repository is prepared for a Flickr8k-style caption dataset. Obtain the dataset from a legitimate source and follow its licence. Put the images and annotation file in the paths documented in `data/README.md`.

The training script currently prepares captions, tokenizer and model architecture. It is a scaffold, not a complete end-to-end training pipeline: image feature extraction/caching, train-validation split, batching, model fitting, and evaluation still need to be implemented for the chosen dataset.

Run the preparation script:

```bash
python -m scripts.train
```

## Run the app

The inference app expects trained artifacts:

- `models/caption_model.keras`
- `models/tokenizer.pkl`

After training and saving compatible artifacts, launch:

```bash
streamlit run app.py
```

Upload a JPG or PNG image and select **Generate caption**.

## Evaluation

For a meaningful experiment, split images (not individual captions) into training, validation and test sets to prevent image leakage. Report suitable caption metrics such as BLEU-1 to BLEU-4, METEOR, and CIDEr, alongside qualitative examples. Compare generated captions with the human reference captions.

## Limitations

- This repository provides a model architecture, preprocessing utilities and app integration scaffold; it does not include pretrained project-specific decoder weights.
- The inference code assumes a decoder trained with the same InceptionV3 feature representation, tokenizer, and sequence length.
- The default maximum sequence length should be updated to match the trained model.
- Captions may omit details, misidentify objects, or describe relationships incorrectly.
- Use appropriately licensed datasets and document their source.

## Future improvements

- Complete the end-to-end training and evaluation pipeline
- Add attention-based visual feature selection
- Compare CNN backbones and decoder architectures
- Add beam search instead of greedy decoding
- Provide qualitative evaluation and metric plots
- Add unit tests and reproducible experiment configuration
