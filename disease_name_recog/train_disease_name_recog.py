from model.biobert_ner import create_model
from data.dataset_preprocessing import fetch_sentences, get_inputs_labels

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import sys
import os
import argparse

try:
    import wandb
except ModuleNotFoundError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

PROJECT_NAME = "NER"
MODEL_NOTES = None
MAX_LENGTH = 75
BATCH_SIZE = 16
EPOCHS = 10
np.random.seed(42)
tf.random.set_seed(42)
DATA_DIR = "../dataloaders/data/disease_name_recog/"


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)

    # Arguments for Project name and Notes on the weights and biases tool
    parser.add_argument("-n", "--notes", type=str, default=MODEL_NOTES, help="Notes about the run")
    parser.add_argument("-p", "--project_name", type=str, default=PROJECT_NAME, help="Main project name")

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    sentences, tags = fetch_sentences(f"{DATA_DIR}train_dev.tsv")
    input_ids, masks, labels, sentence_length = get_inputs_labels(sentences, tags, MAX_LENGTH)
    train_inputs, val_inputs, train_tags, val_tags = train_test_split(input_ids, labels, random_state=42, test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(masks, input_ids, random_state=42, test_size=0.1)

    test_sentences, test_tags = fetch_sentences(f"{DATA_DIR}test.tsv")
    test_input_ids, test_masks, test_tags, test_sentence_length = get_inputs_labels(test_sentences, test_tags, MAX_LENGTH)

    unique_tags = list(set(tag for doc in tags for tag in doc))
    nb_tags = len(unique_tags)

    if args.wandb:
        # initialize wandb logging to your project
        wandb.init(project=args.project_name, notes=args.notes)
        # log all experimental args to wandb
        wandb.config.update(args)
        # ckpt_wandb = tf.keras.callbacks.ModelCheckpoint(f"{wandb.run.dir}/best_model", monitor="val_f1_score_micro", mode='max', verbose=0, save_best_only=True, save_weights_only=True)
        callbacks = [wandb.keras.WandbCallback()]
    else:
        callbacks = None

    model = create_model(nb_tags, MAX_LENGTH)
    model.fit([train_inputs, train_masks], train_tags, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, callbacks=callbacks)

    test_pred = model.predict([test_input_ids, test_masks])

    test_labels = []
    preds = []
    for (y_p, label, sent_len) in zip(test_pred, test_tags, test_sentence_length):
        pred = tf.math.argmax(y_p[:sent_len], -1)
        tru = label[:sent_len]
        test_labels.extend(tru)
        preds.extend(pred)

    print(classification_report(test_labels, preds))


if __name__ == '__main__':
    main()

