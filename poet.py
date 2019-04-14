from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os, pickle
import time
from os import listdir
from os.path import isfile, join
import functools


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]
            ),
            rnn(
                rnn_units,
                return_sequences=True,
                recurrent_initializer="glorot_uniform",
                stateful=True,
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


tf.enable_eager_execution()

text_list = []

data_path = "./data"
datafiles = [
    os.path.join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))
]
print("number of files", len(datafiles))
baits_len = []
for path_to_file in datafiles:
    data = open(path_to_file, "rb").read().decode(encoding="utf-8")
    for line in data.split("\n"):
        line = line.replace("(", "").replace(")", "").replace(r"\\1\u200c\\2", "")
        if len(line) != 0:
            baits_len.append(len(line))
            text_list.append(line)

seq_length = np.percentile(np.array(baits_len), 95)
text = "     ".join(text_list)
# length of text is the number of words in it
print("Length of text: {} words".format(len(text)))
vocab = sorted(["".join(word.split(r"\s")) for word in set(text)])
print("{} unique words".format(len(vocab)))
char2idx = {vocab[i]: int(i) for i in range(len(vocab))}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text], dtype=np.dtype(np.int16))
print("{")
for char, _ in zip(char2idx, range(20)):
    print("  {:4s}: {:3d},".format(repr(char), char2idx[char]))
print("  ...\n}")
print(
    "{} ---- characters mapped to int ---- > {}".format(
        repr(text[:13]), text_as_int[:13]
    )
)

# The maximum length sentence we want for a single input in characters
examples_per_epoch = len(text) // seq_length
# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)
# Batch size
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation="sigmoid")
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(
        example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)"
    )

model.summary()
sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)


example_batch_loss = loss(target_example_batch, example_batch_predictions)
print(
    "Prediction shape: ",
    example_batch_predictions.shape,
    " # (batch_size, sequence_length, vocab_size)",
)
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = "./training_checkpoints"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True
)
EPOCHS = 20
history = model.fit(
    dataset.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=int(steps_per_epoch),
    callbacks=[checkpoint_callback],
)


history_filepath = os.path.join(checkpoint_dir, "trainHistoryDict")
model_filepath = os.path.join(checkpoint_dir, "trainedmodel_50Epoch.h5")
model.save(model_filepath)  # saving the model
with open(history_filepath, "wb") as handle:  # saving the history of the model
    pickle.dump(history.history, handle)
