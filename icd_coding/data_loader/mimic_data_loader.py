import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class MimicDataLoader:
    def __init__(self, data_dir):
        """
        A data-loader for loading all the training data, validation data and test data

        :param data_dir: The MIMIC data folder which contains the files of
                        vocab.csv, TOP_50_CODES.csv, train_50.csv, dev_50.csv, test_50.csv
        """
        self.vocab = load_vocab(f"{data_dir}vocab.csv")
        all_codes = load_vocab(f"{data_dir}TOP_50_CODES.csv")
        self.text_train, codes_train = load_data(f"{data_dir}train_50.csv")
        self.text_dev, codes_dev = load_data(f"{data_dir}dev_50.csv")
        self.text_test, codes_test = load_data(f"{data_dir}test_50.csv")
        self.labels_train, code_vocab = transform_code2vector(codes_train, all_codes)
        self.labels_dev, _ = transform_code2vector(codes_dev, all_codes)
        self.labels_test, _ = transform_code2vector(codes_test, all_codes)

    def get_vocab(self):
        return self.vocab

    def get_train_data(self):
        return self.text_train, self.labels_train

    def get_val_data(self):
        return self.text_dev, self.labels_dev

    def get_test_data(self):
        return self.text_test, self.labels_test


def transform_code2vector(codes, all_codes):
    """
    Transform ICD codes of a patient into a multi-hot vector

    :param
    codes (numpy array of string) : ICD codes of each patient e.g. [["96.71;401.9"],["V58.61;427.31"]]
    :param
    all_codes (numpy array of string): All the target ICD codes, in this case TOP 50 codes e.g. [["96.71"], ["401.9"]...]
    :return:
    ((numpy array), (numpy array)) a multi-hot vector, all the ICD codes
    e.g. [0, 1, 1, 0], [123, 321, 221, 456]
    """
    def split_by_semicolon(inputs):
        return tf.strings.split(inputs, sep=';')

    vectorize_layer_code = TextVectorization(standardize=None,
                                             max_tokens=len(all_codes) + 1,
                                             output_mode='binary',
                                             vocabulary=all_codes,
                                             split=split_by_semicolon)
    model_c = tf.keras.models.Sequential()
    model_c.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model_c.add(vectorize_layer_code)
    code_vector = model_c.predict(codes)
    # Delete PAD
    labels = np.delete(code_vector, 0, -1)
    return labels, vectorize_layer_code.get_vocabulary()


def load_data(filepath):
    data_df = pd.read_csv(filepath)
    text = data_df["TEXT"].to_numpy()
    codes = data_df["LABELS"].to_numpy()
    return text, codes


def load_vocab(filepath):
    voc_df = pd.read_csv(filepath, header=None)
    vocab = sorted(voc_df.iloc[:, 0].tolist())
    return vocab



