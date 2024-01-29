import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from dataloaders.icd_coding_data_loader import ICD_Coding_DataLoader
import time
import importlib
import argparse
try:
    import wandb
except ModuleNotFoundError:
    pass


np.random.seed(42)
tf.random.set_seed(42)

PROJECT_NAME = "ICD_CODING"
MODEL_NOTES = None

DATA_DIR = "../dataloaders/data/icd_coding_data/"
NB_CODES = 30


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
    parser.add_argument("--model_class", type=str, default="CNN")
    parser.add_argument("--test_model", type=str)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--nb_codes", type=str, default=NB_CODES)

    # Get the model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    model_class = _import_class(f"models.{temp_args.model_class}")

    # Get model specific arguments
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    # Arguments for Project name and Notes on the weights and biases tool
    parser.add_argument("-n", "--notes", type=str, default=MODEL_NOTES, help="Notes about the run")
    parser.add_argument("-p", "--project_name", type=str, default=PROJECT_NAME, help="Main project name")

    return parser


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

    dataloader = ICD_Coding_DataLoader(args.data_dir, args.nb_codes)
    vocab = dataloader.get_vocab()
    text_train, labels_train = dataloader.get_train_data()
    text_dev, labels_dev = dataloader.get_val_data()
    text_test, labels_test = dataloader.get_test_data()
    print(f"Number of training set: {text_train.shape[0]}")
    print(f"Number of validation set: {text_dev.shape[0]}")
    print(f"Number of test set: {text_test.shape[0]}")

    if args.use_attention:
        ckpt_filepath = f"./training/{args.model_class}_useAtt_{time.strftime('%b_%d_%H:%M:%S', time.localtime())}/best_model"
    else:
        ckpt_filepath = f"./training/{args.model_class}_{time.strftime('%b_%d_%H:%M:%S', time.localtime())}/best_model"
    scheduler_callbacks = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score_micro', mode='max', patience=6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(ckpt_filepath, monitor="val_f1_score_micro", mode='max', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score_micro', mode='max', factor=0.1, patience=4)

    if args.wandb:
        # initialize wandb logging to your project
        wandb.init(project=args.project_name, notes=args.notes)
        # log all experimental args to wandb
        wandb.config.update(args)
        ckpt_wandb = tf.keras.callbacks.ModelCheckpoint(f"{wandb.run.dir}/best_model", monitor="val_f1_score_micro", mode='max', verbose=0, save_best_only=True, save_weights_only=True)
        callbacks = [early_stopping, reduce_lr, checkpoint, ckpt_wandb, wandb.keras.WandbCallback(monitor="val_f1_score_macro", mode='max', save_weights_only=True)]
    else:
        callbacks = [early_stopping, reduce_lr, checkpoint]

    model_class = _import_class(f"models.{args.model_class}")
    model = model_class(vocab=vocab, args=args)

    if args.test_model:
        model = model.build_model(labels_test[0].shape[0])
        model.load_weights(args.test_model)
        print(model.summary())
        test_predict = (np.asarray(model.predict(text_test))).round()
        test_f1_macro = f1_score(labels_test, test_predict, average='macro')
        test_f1_micro = f1_score(labels_test, test_predict, average='micro')
        print("test_f1_macro: " + f"{test_f1_macro:.3f}")
        print("test_f1_micro: " + f"{test_f1_micro:.3f}")
    else:
        model.fit(text_train, labels_train, validation_data=(text_dev, labels_dev), callbacks=callbacks)
        model.load(ckpt_filepath)
        test_predict = (np.asarray(model.predict(text_test))).round()
        test_predict_prob = model.predict(text_test)
        test_f1_macro = f1_score(labels_test, test_predict, average='macro')
        test_f1_micro = f1_score(labels_test, test_predict, average='micro')
        test_auc_macro = roc_auc_score(labels_test, test_predict_prob, average='macro')
        test_auc_micro = roc_auc_score(labels_test, test_predict_prob, average='micro')
        test_p_macro = precision_score(labels_test, test_predict, average='macro')
        test_p_micro = precision_score(labels_test, test_predict, average='micro')
        test_r_macro = recall_score(labels_test, test_predict, average='macro')
        test_r_micro = recall_score(labels_test, test_predict, average='micro')

        print(f"test_f1_macro: {test_f1_macro:.3f}")
        print(f"test_f1_micro: {test_f1_micro:.3f}")
        print(f"test_auc_macro: {test_auc_macro:.3f}")
        print(f"test_auc_micro: {test_auc_micro:.3f}")
        print(f"test_precision_macro: {test_p_macro:.3f}")
        print(f"test_precision_micro: {test_p_micro:.3f}")
        print(f"test_recall_macro: {test_r_macro:.3f}")
        print(f"test_recall_micro: {test_r_micro:.3f}")

        prec_top5 = tf.keras.metrics.Precision(top_k=5)
        prec_top8 = tf.keras.metrics.Precision(top_k=8)
        prec_top5.update_state(labels_test, test_predict)
        prec_top8.update_state(labels_test, test_predict)
        print(f"test_precision_top5: {prec_top5.result().numpy()}")
        print(f"test_precision_top8: {prec_top8.result().numpy()}")


if __name__ == '__main__':
    main()
