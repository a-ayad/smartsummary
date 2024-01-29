import os

import numpy as np
import tensorflow as tf

from utils.misc import load_labels
from sklearn.model_selection import train_test_split

import time
import importlib
import argparse
try:
    import wandb
except ModuleNotFoundError:
    pass


np.random.seed(42)
tf.random.set_seed(42)

PROJECT_NAME = "Time Series Classification"
MODEL_NOTES = None


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
    parser.add_argument("--model_class", type=str, default="LSTM")
    parser.add_argument("--only_lab", type=bool, default=True)

    # Get the model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    model_class = _import_class(f"models.{temp_args.model_class}")

    # Get model specific arguments
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    parser.add_argument("-m", "--notes", type=str, default=MODEL_NOTES, help="Notes about the training_tsc run")
    parser.add_argument("-p", "--project_name", type=str, default=PROJECT_NAME, help="Main project name")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    '''
    python train_tsc.py  --model_class=LSTM --lstm_unit=256 --dropout=0.3 --learning_rate=0.001
    '''
    """
    parser = _setup_parser()
    args = parser.parse_args()

    ckpt_filepath = f"./training/{args.model_class}_{time.strftime('%b_%d_%H:%M:%S', time.localtime())}/best_model"
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_filepath, monitor="val_loss", mode='min', verbose=0,
                                                    save_best_only=True, save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=4)

    if args.wandb:
        # initialize wandb logging to your project
        wandb.init(project=args.project_name, notes=args.notes)
        # log all experimental args to wandb
        wandb.config.update(args)
        callbacks = [early_stopping, checkpoint, reduce_lr, wandb.keras.WandbCallback()]
    else:
        callbacks = [early_stopping, checkpoint, reduce_lr]

    X = np.load('../dataloaders/data/tsc_data/features.npy')
    y, FEATURE_COLUMNS = load_labels()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    INPUT_SHAPE = X_train.shape[1:]
    N_FEATURES = y.shape[1]
    model_class = _import_class(f"models.{args.model_class}")
    model = model_class(INPUT_SHAPE, N_FEATURES, args=args)
    model.fit(X_train, y_train, X_test, y_test, callbacks=callbacks)

    model.load(ckpt_filepath)
    model.predict(X_test, y_test, FEATURE_COLUMNS)


if __name__ == '__main__':
    main()