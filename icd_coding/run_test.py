import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from data_loader import utils
import importlib
import argparse

import wandb
from wandb.keras import WandbCallback

np.random.seed(42)
tf.random.set_seed(42)

PROJECT_NAME = "ICD_CODING"
MODEL_NOTES = "SWEEP"
MAX_LENGTH = 2500
LEARNING_RATE = 0.003
BATCH_SIZE = 32
EPOCHS = 100


def _import_class(module_and_class_name: str):
    """Import class from a module, e.g. 'models.LSTM' """
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--model_class", type=str, default="CNNwLSTM")
    parser.add_argument("--max_len", type=int, default=MAX_LENGTH)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    ####TODO
    parser.add_argument("--ba", type=int, default=BATCH_SIZE)
    parser.add_argument("--ep", type=int, default=EPOCHS)

    # Get the model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    model_class = _import_class(f"models.{temp_args.model_class}")

    # Get model specific arguments
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    parser.add_argument("-n", "--notes", type=str, default=MODEL_NOTES, help="Notes about the run")
    parser.add_argument("-p", "--project_name", type=str, default=PROJECT_NAME, help="Main project name")

    return parser


def transform_code2vector(code, all_codes):

    def split_by_semicolon(inputs):
        return tf.strings.split(inputs, sep=';')

    vectorize_layer_code = TextVectorization(standardize=None,
                                             max_tokens=len(all_codes)+1,
                                             output_mode='binary',
                                             vocabulary=all_codes,
                                             split=split_by_semicolon)
    model_c = tf.keras.models.Sequential()
    model_c.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model_c.add(vectorize_layer_code)
    code_vector = model_c.predict(code)
    # Delete PAD
    labels = np.delete(code_vector, 0, -1)
    return labels, vectorize_layer_code.get_vocabulary()


def vector2code(vector, all_codes):
    # t = [all_codes[i + 1] for i, prob in enumerate(pr[0]) if prob > 0.5]
    codes = [all_codes[i] for i, code in enumerate(vector) if int(code) == 1]
    return codes


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    vocab = utils.load_vocab("mimicdata/vocab.csv")
    all_codes = utils.load_vocab("mimicdata/TOP_50_CODES.csv")
    text_train, codes_train = utils.load_data("mimicdata/train_50.csv")
    text_dev, codes_dev = utils.load_data("mimicdata/dev_50.csv")
    labels_train, code_vocab = transform_code2vector(codes_train, all_codes)
    labels_dev, _ = transform_code2vector(codes_dev, all_codes)

    scheduler_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score_macro', mode='max', patience=10)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('./training/epoch:{epoch}-f1:{val_f1_score_macro:.3f}_' + args.model_class,
                                                    monitor="val_f1_score_macro",
                                                    mode='max',
                                                    verbose=0,
                                                    save_best_only=True)
    if args.wandb:
        # initialize wandb logging to your project
        wandb.init(project=args.project_name, notes=args.notes)
        # log all experimental args to wandb
        wandb.config.update(args)
        callbacks = [early_stopping, checkpoint, WandbCallback(monitor="val_f1_score_macro", mode='max')]
    else:
        callbacks = [early_stopping, checkpoint]

    vectorization_layer = TextVectorization(standardize="lower_and_strip_punctuation",
                                            max_tokens=len(vocab) + 2,
                                            output_mode='int',
                                            output_sequence_length=args.max_length)
    vocab_dataset = tf.data.Dataset.from_tensor_slices(vocab)
    vectorization_layer.adapt(vocab_dataset)


    train_dataset = tf.data.Dataset.from_tensor_slices((text_train, labels_train)).batch(args.ba)
    train_dataset = train_dataset.map(lambda x, y: (vectorization_layer(x), y))
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    dev_dataset = tf.data.Dataset.from_tensor_slices((text_dev, labels_dev)).batch(args.ba)
    dev_dataset = dev_dataset.map(lambda x, y: (vectorization_layer(x), y))
    dev_dataset = dev_dataset.prefetch(tf.data.AUTOTUNE)

    nb_classes = labels_train[0].shape[0]
    model_class = _import_class(f"models.{args.model_class}")
    model = model_class(nb_classes, len(vectorization_layer.get_vocabulary()), args=args)
    '''
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
    x = vectorization_layer(text_input)
    outputs = model(x)
    model = tf.keras.Model(text_input, outputs)
    '''

    f1_macro = tfa.metrics.F1Score(num_classes=nb_classes, threshold=0.5, average='macro', name='f1_score_macro')
    f1_micro = tfa.metrics.F1Score(num_classes=nb_classes, threshold=0.5, average='micro', name='f1_score_micro')
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  metrics=[tf.keras.metrics.AUC(multi_label=True),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision(),
                           f1_macro,
                           f1_micro])

    model.fit(train_dataset,
              validation_data=dev_dataset,
              epochs=1,
              callbacks=callbacks)
    print(model.summary())
    results = model.evaluate(text_dev[:5], labels_dev[:5])
    # print(results)
    '''
    loaded_model = tf.keras.models.load_model(filepath)
    test_model = tf.keras.models.Model(inputs=loaded_model.input, outputs=[loaded_model.output, loaded_model.get_layer('attention').output[0]])
    op = test_model.predict(['test_text'])
    # op[0] -> output of the model
    # op[1] -> attention weights
    '''


if __name__ == '__main__':
    main()
