#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from reading_comprehension_dataset import ReadingComprehensionDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--lr", default=1e-5, type=float,  help="Learning Rate.")
parser.add_argument("--epsilon", default=1e-08, type=float,  help="Epsilon.")
parser.add_argument("--clipnorm", default=1, type=float,  help="Clipnorm.")


class Model(tf.keras.Model):
    def __init__(self, electra_model, args):

        input_layer = tf.keras.layers.Input([None], dtype=tf.int64, ragged=True)
        mask = tf.sequence_mask(input_layer.row_lengths())
        hidden_layer = electra_model(input_layer.to_tensor(), attention_mask=mask)[0]

        start = tf.keras.layers.Dense(1, activation=None)(hidden_layer)
        start= tf.squeeze(start)
        start = tf.keras.layers.Softmax()(start)
        end = tf.keras.layers.Dense(1, activation=None)(hidden_layer)
        end = tf.keras.layers.Softmax()(tf.squeeze(end, axis=-1))

        super().__init__(inputs=input_layer, outputs={"start":start,"end":end})

        lr= tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0001, alpha=0.001, decay_steps= 4000)

        epsilon=1e-08
        clipnorm=1
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon, clipnorm=clipnorm),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        )

def create_dataset_train(dataset, tokenizer):
    inputs = []
    outputs = []

    for paragraph in dataset.paragraphs:
        context = paragraph["context"]
        context_encoded = tokenizer.encode(context)
        qas = paragraph["qas"]
        for qa in qas:
            qa_encoded = tokenizer.encode(qa["question"])
            inp = context_encoded+qa_encoded[1:]
            if len(inp)>512:
                context_encoded = context_encoded[:512-len(inp)-1] + [context_encoded[-1]]
                inp = context_encoded+qa_encoded[1:]

            for answer in qa["answers"]:
                answer_ = answer["text"]
                if len(answer_)==0:
                    continue

                offset = answer["start"]
                token_obj = tokenizer(context)
                start = token_obj.char_to_token(offset)
                end_offset = offset + len(answer_) - 1
                end = token_obj.char_to_token(end_offset)
                if end != None and start != None:

                 if end > len(context_encoded)-2 or start > len(context_encoded)-2:
                    continue

                 start_end=(tf.constant(start),tf.constant(end))

                 inputs.append(inp)
                 outputs.append(start_end)
                 #contexts.append(context)
                
    data_tensors = (tf.ragged.constant(inputs), outputs)
    dataset = tf.data.Dataset.from_tensor_slices(data_tensors)
    dataset = dataset.map(lambda x,y: (x, {"start":y[0], "end":y[1]}))
    

    dataset = dataset.shuffle(len(dataset), seed=args.seed)
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

   
    return dataset

def create_dataset_test(dataset, tokenizer):
    inputs = []
    outputs = []
    contexts = []

    for paragraph in dataset.paragraphs:
        context = paragraph["context"]
        context_encoded = tokenizer.encode(context)
        qas = paragraph["qas"]
        for qa in qas:
            qa_encoded = tokenizer.encode(qa["question"])
            inp = context_encoded+qa_encoded[1:]
            if len(inp)>512:
                context_encoded = context_encoded[:512-len(inp)-1] + [context_encoded[-1]]
                inp = context_encoded+qa_encoded[1:]

            
            inputs.append(inp)
            contexts.append(context)

            for answer in qa["answers"]:
                answer_ = answer["text"]
                if len(answer_)==0:
                    continue

                offset = answer["start"]
                token_obj = tokenizer(context)
                start = token_obj.char_to_token(offset)
                end_offset = offset + len(answer_) - 1
                end = token_obj.char_to_token(end_offset)
                #print(start)
                if end != None and start != None:

                 if end > len(context_encoded)-2 or start > len(context_encoded)-2:
                    continue

                 start_end=(tf.constant(start),tf.constant(end))

                 inputs.append(inp)
                 outputs.append(start_end)
                 contexts.append(context)

                
    data_tensors = tf.ragged.constant(inputs)
    dataset = tf.data.Dataset.from_tensor_slices(data_tensors)

    #dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, contexts
   

def create_dataset(dataset, name, tokenizer):
    inputs = []
    outputs = []
    contexts = []

    for paragraph in dataset.paragraphs:
        context = paragraph["context"]
        context_encoded = tokenizer.encode(context)
        qas = paragraph["qas"]
        for qa in qas:
            qa_encoded = tokenizer.encode(qa["question"])
            inp = context_encoded+qa_encoded[1:]
            if len(inp)>512:
                context_encoded = context_encoded[:512-len(inp)-1] + [context_encoded[-1]]
                inp = context_encoded+qa_encoded[1:]

            if name == "test":
                inputs.append(inp)
                contexts.append(context)

            for answer in qa["answers"]:
                answer_ = answer["text"]
                if len(answer_)==0:
                    continue

                offset = answer["start"]
                token_obj = tokenizer(context)
                start = token_obj.char_to_token(offset)
                end_offset = offset + len(answer_) - 1
                end = token_obj.char_to_token(end_offset)
                #print(start)
                if end != None and start != None:

                 if end > len(context_encoded)-2 or start > len(context_encoded)-2:
                    continue

                 start_end=(tf.constant(start),tf.constant(end))

                 inputs.append(inp)
                 outputs.append(start_end)
                 contexts.append(context)
                
    if name == "train" or name == "dev":
        data_tensors = (tf.ragged.constant(inputs), outputs)
        dataset = tf.data.Dataset.from_tensor_slices(data_tensors)
        dataset = dataset.map(lambda x,y: (x, {"start":y[0], "end":y[1]}))
    else:
        data_tensors = tf.ragged.constant(inputs)
        dataset = tf.data.Dataset.from_tensor_slices(data_tensors)

    dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if name == 'test':
        return dataset, contexts
    else:
        return dataset

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

    # Load the pre-trained RobeCzech model
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.TFAutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data
    dataset = ReadingComprehensionDataset()

    # TODO: Create the model and train it
    model = Model(robeczech,args)
    train = create_dataset_train(dataset.train, tokenizer=tokenizer)
    dev= create_dataset_train(dataset.dev,tokenizer)
    
    
    model.fit(train, epochs=args.epochs, validation_data=dev)

    test, context= create_dataset_test(dataset.test, tokenizer=tokenizer)
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        iter = 0
        for t in test:
            predictions = model.predict([t])
            starts = predictions["start"]
            ends = predictions["end"]

            for start_id in range(len(starts)):
                start = np.argmax(starts[start_id][1:])+1
                end = min(start + np.argmax(ends[start_id][start:]), len(tokenizer.encode(context[iter]))-2)
                encoded = tokenizer(context[id])

                if start <= end:
                    answer = context[iter][encoded.token_to_chars(start).start: encoded.token_to_chars(end).end]
                else:
                    answer = ""

                iter += 1

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
