
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.dataset_preprocessing import fetch_sentences, get_inputs_labels
from model.biobert_ner import sparse_categorical_accuracy_masked, sparse_crossentropy_masked
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)
tokenizer = BertTokenizer('biobert_v1.1_pubmed/vocab.txt')
MAX_LENGTH = 125
biobert_model = tf.keras.models.load_model('model/biobert_ner',
                                           custom_objects={'sparse_categorical_accuracy_masked': sparse_categorical_accuracy_masked,
                                                           'sparse_crossentropy_masked': sparse_crossentropy_masked})

FILEPATH = "../dataloaders/data/disease_name_recog/train_dev.tsv"
sentences, tags = fetch_sentences(FILEPATH)
input_ids, masks, labels, sentence_length = get_inputs_labels(sentences, tags, MAX_LENGTH)
train_inputs, val_inputs, train_tags, val_tags = train_test_split(input_ids, labels, random_state=42, test_size=0.1)
train_masks, val_masks, train_sent_len, val_sent_len = train_test_split(masks, sentence_length, random_state=42, test_size=0.1)

y_pred = biobert_model.predict([val_inputs, val_masks])

val_labels = []
preds = []
for (y_p, label, sent_len) in zip(y_pred, val_tags, val_sent_len):
    pred = tf.math.argmax(y_p[:sent_len], -1)
    tru = label[:sent_len]
    val_labels.extend(tru)
    preds.extend(pred)

print(classification_report(val_labels, preds))