from model.biobert_ner import create_model
from data.dataset_preprocessing import fetch_sentences, get_inputs_labels

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


MAX_LENGTH = 120
np.random.seed(42)
tf.random.set_seed(42)


def main():
    file = "data/train_dev.tsv"
    sentences, tags = fetch_sentences(file)
    input_ids, masks, labels = get_inputs_labels(sentences, tags, MAX_LENGTH)
    train_inputs, val_inputs, train_tags, val_tags = train_test_split(input_ids, labels, random_state=42, test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(masks, input_ids, random_state=42, test_size=0.1)
    unique_tags = list(set(tag for doc in tags for tag in doc))
    nb_tags = len(unique_tags)
    model = create_model(nb_tags, MAX_LENGTH)
    model.fit([train_inputs, train_masks], train_tags, epochs=10, batch_size=16, validation_split=0.1)


if __name__ == '__main__':
    main()

