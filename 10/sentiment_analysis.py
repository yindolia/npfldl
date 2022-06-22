#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from unicodedata import bidirectional
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
try:
    import transformers
except Exception:
    raise RuntimeError("You need to install the `transformers` package")

from text_classification_dataset import TextClassificationDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small")
    eleczech.trainable = False

    # TODO: Load the data. Consider providing a `tokenizer` to the
    # constructor of the TextClassificationDataset.
    facebook = TextClassificationDataset("czech_facebook", tokenizer=tokenizer.encode)

    # TODO: Create the model and train it
    def train_map(example):
        tokens=example["tokens"]
        labels=facebook.train.label_mapping(example["labels"])
        return (tokens, labels)
    
    def test_map(example):
        return example["tokens"]
    def prepare_test(name):
        dataset = getattr(facebook, name).dataset
        dataset = dataset.map(test_map)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    def prepare_train(name):
        dataset = getattr(facebook, name).dataset
        dataset = dataset.map(train_map)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = prepare_train("train"), prepare_train("dev")
    test= prepare_test("test")

    input= tf.keras.layers.Input(shape=[None],ragged=True, dtype=tf.int32)
    row_lengths= input.row_lengths()
    eleczech_embeddings= eleczech(input.to_tensor(), attention_mask= tf.sequence_mask(row_lengths) )
    embeddings_layer= tf.RaggedTensor.from_tensor(eleczech_embeddings.last_hidden_state, lengths=row_lengths)

    cell= tf.keras.layers.GRU(512, return_sequences= False)
    bidirectional_layer= tf.keras.layers.Bidirectional(cell, merge_mode='sum')(embeddings_layer)
    #rnn_layer= tf.keras.layers.GRU(args.rnn_dim, return_sequences=True)(bidirectional_layer)

    output= tf.keras.layers.Dense(3, activation= tf.nn.softmax)(bidirectional_layer)

    model = tf.keras.Model(input, output)
    steps=args.epochs * facebook.train.size / args.batch_size

    lr= tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0005, alpha=0.001, decay_steps= 3000
    )
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")])

    model.fit(train, epochs=args.epochs, validation_data=dev)


    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the tags on the test set.
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
