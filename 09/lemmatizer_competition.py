#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=1000, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self.source_mapping = train.forms.char_mapping
        self.target_mapping = train.lemmas.char_mapping
        self.target_mapping_inverse = type(self.target_mapping)(
            vocabulary=self.target_mapping.get_vocabulary(), invert=True)
        self.target_vocabulary_size= self.target_mapping.vocabulary_size()
        # TODO(lemmatizer_noattn): Define
        # - `self.source_embedding` as an embedding layer of source chars into `args.cle_dim` dimensions
        self.source_embedding = tf.keras.layers.Embedding(input_dim=self.source_mapping.vocabulary_size(), output_dim=args.cle_dim)
        # TODO: Define
        # - `self.source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning **whole sequences**,
        #   summing opposite directions
        self.source_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_dim, return_sequences=True), merge_mode='sum')

        # TODO(lemmatizer_noattn): Then define
        # - `self.target_embedding` as an embedding layer of target chars into `args.cle_dim` dimensions
        # - `self.target_rnn_cell` as a GRUCell with `args.rnn_dim` units
        # - `self.target_output_layer` as a Dense layer into as many outputs as there are unique target chars
        self.target_embedding = tf.keras.layers.Embedding(input_dim=self.target_mapping.vocabulary_size(), output_dim=args.cle_dim)
        self.target_rnn_cell = tf.keras.layers.GRUCell(args.rnn_dim)
        self.target_output_layer = tf.keras.layers.Dense(self.target_mapping.vocabulary_size(), activation=None)
        # TODO: Define
        # - `self.attention_source_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_state_layer` as a Dense layer with `args.rnn_dim` outputs
        # - `self.attention_weight_layer` as a Dense layer with 1 output
        self.attention_source_layer = tf.keras.layers.Dense(args.rnn_dim, activation=None)
        self.attention_state_layer = tf.keras.layers.Dense(args.rnn_dim, activation=None)
        self.attention_weight_layer = tf.keras.layers.Dense(1, activation=None)

        # Compile the model
        initial_learning_rate=0.01
        lr=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate, 10000, alpha=0.0, name=None
            )
        self.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    class DecoderTraining(tfa.seq2seq.BaseDecoder):
        def __init__(self, lemmatizer, *args, **kwargs):
            self.lemmatizer = lemmatizer
            super().__init__.__wrapped__(self, *args, **kwargs)

        @property
        def batch_size(self):
            # TODO(lemmatizer_noattn): Return the batch size of `self.source_states` as a *scalar* number;
            # use `tf.shape` to get the full shape and then extract the batch size.
            return tf.shape(self.source_states)[0]
        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning
            #   tf.TensorShape(number of logits of each output element [lemma character])
            #raise NotImplementedError()
            return tf.float32

        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Return the type of the decoder output (so the type of the
            # produced logits).]
            #with tf.init_scope():
            return tf.TensorShape(self.lemmatizer.target_vocabulary_size)
                #return tf.TensorShape(self.lemmatizer.target_mapping.vocabulary_size())

            raise NotImplementedError()

        def with_attention(self, inputs, states):
            
            source_states = self.lemmatizer.attention_source_layer(self.source_states)
            states = self.lemmatizer.attention_state_layer(states)
            sum = source_states + tf.expand_dims(states, axis=1)
            sum = tf.tanh(sum)
            sum = self.lemmatizer.attention_weight_layer(sum)
            weights = tf.nn.softmax(sum, axis=1)
            attention = tf.math.multiply(self.source_states, weights)
            attention = tf.reduce_sum(attention, axis=1)
            return tf.concat([inputs, attention], axis=1)

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            self.source_states, self.targets = layer_inputs

            # TODO(lemmatizer_noattn): Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
            finished = tf.fill([self.batch_size], False)

            # TODO(lemmatizer_noattn): Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
            # embedded using self.lemmatizer.target_embedding
            inputs = self.lemmatizer.target_embedding(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))

            # TODO: Define `states` as the representation of the first character
            # in `source_states`. The idea is that it is most relevant for generating
            # the first letter and contains all following characters via the backward RNN.
            states = self.source_states[:,0,:]

            # TODO: Pass `inputs` through `self.with_attention(inputs, states)`.
            inputs= self.with_attention(inputs=inputs,states=states)
            return finished, inputs, states

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            (outputs, [states]) = self.lemmatizer.target_rnn_cell(inputs, [states])

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)

            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding `time`-th chars from `self.targets`.
            next_inputs = self.lemmatizer.target_embedding(self.targets[:, time])

            # TODO(lemmatizer_noattn): Define `finished` as a vector of booleans; True if the corresponding
            # `time`-th char from `self.targets` is `MorphoDataset.Factor.EOW`, False otherwise.
            finished = self.targets[:, time] == MorphoDataset.Factor.EOW

            # TODO: Pass `next_inputs` through `self.with_attention(next_inputs, states)`.
            next_inputs= self.with_attention(inputs=next_inputs,states= states)
            return outputs, states, next_inputs, finished

    class DecoderPrediction(DecoderTraining):
        @property
        def output_size(self):
            # TODO(lemmatizer_noattn): Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning a suitable
            # `tf.TensorShape` representing a *scalar* element, because we are producing
            # lemma character indices during prediction.
            return tf.TensorShape([])

            #raise NotImplementedError()
        @property
        def output_dtype(self):
            # TODO(lemmatizer_noattn): Return the type of the decoder output (i.e., target lemma character indices).
            return tf.int32

            #raise NotImplementedError()

        def initialize(self, layer_inputs, initial_state=None, mask=None):
            # Use `initialize` from the `DecoderTraining`, passing None as `targets`.
            return super().initialize([layer_inputs, None], initial_state)

        def step(self, time, inputs, states, training):
            # TODO(lemmatizer_noattn): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            (outputs, [states]) = self.lemmatizer.target_rnn_cell(inputs,[states])

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)

            # TODO(lemmatizer_noattn): Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
            # `output_type=tf.int32` parameter.
            outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)

            # TODO(lemmatizer_noattn): Define `next_inputs` by embedding the `outputs`
            next_inputs = self.lemmatizer.target_embedding(outputs)

            # TODO(lemmatizer_noattn): Define `finished` as a vector of booleans; True if the corresponding
            # prediction in `outputs` is `MorphoDataset.Factor.EOW`, False otherwise.
            finished = (outputs == MorphoDataset.Factor.EOW)

            # TODO(DecoderTraining): Pass `next_inputs` through `self.with_attention(next_inputs, states)`.
            next_inputs= self.with_attention(inputs=next_inputs,states= states)

            return outputs, states, next_inputs, finished

    # If `targets` is given, we are in the teacher forcing mode.
    # Otherwise, we run in autoregressive mode.
    def call(self, inputs, targets=None):
        # Forget about sentence boundaries and instead consider
        # all valid form-lemma pairs as independent batch examples.
        #
        # Then, split the given forms into character sequences and map then
        # to their indices.
        source_charseqs = inputs.values
        source_charseqs = tf.strings.unicode_split(source_charseqs, "UTF-8")
        source_charseqs = self.source_mapping(source_charseqs)
        if targets is not None:
            # The targets are already mapped sequences of characters, so only
            # drop the sentence boundaries, and convert to a dense tensor
            # (the EOW correctly indicate end of lemma).
            target_charseqs = targets.values
            target_charseqs = target_charseqs.to_tensor()

        # TODO(lemmatizer_noattn): Embed source_charseqs using `source_embedding`
        embedding=self.source_embedding(source_charseqs)

        # TODO: Run source_rnn on the embedded sequences, returning outputs in `source_states`.
        # However, convert the embedded sequences from a RaggedTensor to a dense Tensor first,
        # i.e., call the `source_rnn` with
        #   (source_embedded.to_tensor(), mask=tf.sequence_mask(source_embedded.row_lengths()))
        embed_lengths= embedding.row_lengths()
        embedding= embedding.to_tensor()
        source_states = self.source_rnn(embedding, mask=tf.sequence_mask(embed_lengths))

        # Run the appropriate decoder. Note that the outputs of the decoders
        # are exactly the outputs of `tfa.seq2seq.dynamic_decode`.
        if targets is not None:
            # TODO(lemmatizer_noattn): Create a self.DecoderTraining by passing `self` to its constructor.
            # Then run it on `[source_states, target_charseqs]` input,
            # storing the first result in `output` and the third result in `output_lens`.
            decoder_train=self.DecoderTraining(self)
            output, _, output_lens= decoder_train([source_states, target_charseqs])
            
            #raise NotImplementedError()
        else:
            # TODO(lemmatizer_noattn): Create a self.DecoderPrediction by using:
            # - `self` as first argument to its constructor
            # - `maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32)`
            #   as another argument, which indicates that the longest prediction
            #   must be at most 10 characters longer than the longest input.
            decoder_predict= self.DecoderPrediction(self, maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, 
                                                    dtype=tf.int32))
            
            # Then run it on `source_states`, storing the first result in `output`
            # and the third result in `output_lens`. Finally, because we do not want
            # to return the `[EOW]` symbols, subtract one from `output_lens`.
            
            
            #raise NotImplementedError()
            output,second, output_lens= decoder_predict(source_states)
            output_lens= output_lens-1
       
        # Reshape the output to the original matrix of lemmas
        # and explicitly set mask for loss and metric computation.
        output = tf.RaggedTensor.from_tensor(output, output_lens)
        output = inputs.with_values(output)
        return output

    def train_step(self, data):
        x, y = data

        # Convert `y` by splitting characters, mapping characters to ids using
        # `self.target_mapping` and finally appending `MorphoDataset.Factor.EOW`
        # to every sequence.
        y_targets = self.target_mapping(tf.strings.unicode_split(y.values, "UTF-8"))
        y_targets = tf.concat(
            [y_targets, tf.fill([y_targets.bounding_shape(0), 1],
                                tf.constant(MorphoDataset.Factor.EOW, tf.int64))], axis=-1)
        y_targets = y.with_values(y_targets)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y_targets, training=True)
            loss = self.compute_loss(x, y_targets.flat_values, y_pred.flat_values)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(x)
        self.compiled_metrics.update_state(tf.ones_like(y, dtype=tf.int32), tf.cast(y_pred == y, tf.int32))
        return {m.name: m.result() for m in self.metrics if m.name != "loss"}


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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt_lemmas", add_bow_eow=True)
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # TODO: Create the model and train it
    model = Model(args, morpho.train)

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda example: (example["forms"], example["lemmas"]))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                     callbacks=[model.tb_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "lemmatizer_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set; update the following prediction
        # command if you use other output structre than in lemmatizer_noattn.
        predictions = model.predict(test)
        for sentence in predictions:
            for word in sentence:
                print(word.numpy().decode("utf-8"), file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
