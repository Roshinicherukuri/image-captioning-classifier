from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add

def build_caption_model(vocab_size, max_length, feature_dim=2048):
    """Build a CNN-feature + LSTM language decoder."""
    image_input = Input(shape=(feature_dim,), name="image_features")
    image_branch = Dropout(0.4)(image_input)
    image_branch = Dense(256, activation="relu")(image_branch)

    sequence_input = Input(shape=(max_length,), name="caption_sequence")
    text_branch = Embedding(vocab_size, 256, mask_zero=True)(sequence_input)
    text_branch = Dropout(0.4)(text_branch)
    text_branch = LSTM(256)(text_branch)

    merged = Add()([image_branch, text_branch])
    output = Dense(vocab_size, activation="softmax")(merged)
    return Model(inputs=[image_input, sequence_input], outputs=output)
