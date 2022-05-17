#!/usr/bin/env python3
import argparse
import datetime
import functools
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from homr_dataset import HOMRDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--rnn_dim", default=256, type=int, help="RNN cell dimension.")

HEIGHT= 64


def rnn_block(inputs):
    layer= tf.keras.layers.GRU(args.rnn_dim, return_sequences=True)
    #inputs= tf.cast(inputs, tf.int32)
    bid= tf.keras.layers.Bidirectional(layer, merge_mode='sum')(tf.RaggedTensor.from_tensor(inputs))
    return bid

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        input= tf.keras.Input(type_spec=tf.RaggedTensorSpec(shape=[None,HEIGHT, None,1],
                                        dtype=tf.float32, ragged_rank=1))
    
        rag_length= input.row_lengths()
        cb= conv_block(input.to_tensor(), 32)
        cb= conv_block(cb,64)
        cb= conv_block(cb,96)
        cb= conv_block(cb,128)

        cb = tf.transpose(cb,perm=[0,2,1,3])
        cb= tf.reshape(cb, [-1,tf.shape(cb)[1],tf.shape(cb)[2]*tf.shape(cb)[3]])
        #cb= tf.squeeze()

        lstm_layer= tf.keras.layers.GRU(args.rnn_dim, return_sequences=True)
        bid= tf.keras.layers.Bidirectional(lstm_layer, merge_mode='sum')(tf.RaggedTensor.from_tensor(cb))
        #bid= tf.keras.layers.Bidirectional(lstm_layer, merge_mode='sum')(bid)
        #for _ in range(3):
          #  cb= rnn_block(cb)
        rnn_layer= tf.keras.layers.GRU(args.rnn_dim, return_sequences=True)(bid)
        #fclayer= tf.keras.layers.Dense(args.rnn_dim, activation='relu')(cb)
        logits = tf.keras.layers.Dense(1+len(HOMRDataset.MARKS), activation=None)(rnn_layer)

        super().__init__(inputs=input, outputs=logits)
        
        lr=tf.keras.optimizers.schedules.CosineDecay(
            0.001, 8000, alpha=0.0001, name=None
            )
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                     loss=self.ctc_loss,
                     metrics=[HOMRDataset.EditDistanceMetric()])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        gold_labels= tf.cast( gold_labels.to_sparse(), dtype=tf.int32)
        logit_length= tf.cast(logits.row_lengths(), dtype=tf.int32)
        logits= logits.to_tensor()
        highest_index= len(HOMRDataset.MARKS)
        loss= tf.nn.ctc_loss(gold_labels, logits, label_length=None, logit_length=logit_length,
                            logits_time_major=False, blank_index=highest_index)

        loss= tf.reduce_mean(loss)
        return loss
       

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        #assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        logits_length=tf.cast(logits.row_lengths(), dtype=tf.int32 )
        logits=logits.to_tensor()
        logits= tf.transpose(logits, [1,0,2])
        #predictions, score= tf.nn.ctc_beam_search_decoder(logits, logits_length, beam_width=1)
        predictions, score= tf.nn.ctc_greedy_decoder(logits, sequence_length=logits_length)

        predictions = tf.RaggedTensor.from_sparse(predictions[0])

        #assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)

def prepare_dataset(example):
    image_len= tf.shape(example["image"])[1]
    example["image"]= tf.image.resize(example["image"],[HEIGHT,image_len])

    return example

def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x= tf.keras.layers.MaxPooling2D()(x)
    return x
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

    # Load the data
    homr = HOMRDataset()


    #train = homr.train
    #dev= homr.dev
    #test= homr.test

    #train=train.map(prepare_dataset)
    #dev= dev.map(prepare_dataset)

    def create_dataset(name):
        def prepare_example(example):

            image, output= example['image'], example['marks']
            input= tf.image.resize(image,[HEIGHT,tf.shape(image)[1]])
            return input, output
            #raise NotImplementedError()

        dataset = getattr(homr, name).map(prepare_example)
        dataset = dataset.shuffle(20*args.batch_size, seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")
    # TODO: Create the model and train it

    '''input= tf.keras.Input(type_spec=tf.RaggedTensorSpec(shape=[args.batch_size,128,None,1],ragged_rank=1, dtype=tf.float32))
    rag_length= input.row_lengths()
    ten_input= input.to_tensor()
    #input= input.to_tensor()
    #tf.print(tf.shape(input.to_tensor()))
    x = tf.keras.layers.Conv2D(32, 3, padding="same",input_shape=[-1,128,None,1])(ten_input)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    cb= conv_block(input.to_tensor(), 32)
    cb= conv_block(cb,64)
    cb= conv_block(cb,64)
    #cb= conv_block(cb,64)
    cb = tf.transpose(cb,perm=[0,2,1,3])
    cb= tf.reshape(cb, [-1,rag_length[0],tf.shape(cb)[2]*tf.shape(cb)[3]])

    lstm_layer= tf.keras.layers.LSTM(args.rnn_dim, return_sequences=True)
    bid= tf.keras.layers.Bidirectional(lstm_layer, merge_mode='sum')(cb)

    logits = tf.keras.layers.Dense(len(homr.MARKS), activation=None)(bid)
    #logit_length= tf.cast(logits.row_lengths(), dtype=tf.int32)
    model = tf.keras.Model(inputs=input, outputs=logits)
    
    model.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.nn.ctc_loss,
                     metrics=[HOMRDataset.EditDistanceMetric()])'''
    
    model=Model(args)
    
    model.fit(train, epochs=args.epochs, validation_data=dev)
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        #predictions = model.predic
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

