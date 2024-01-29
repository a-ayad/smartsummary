import csv

import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# tokenizer = BertTokenizer('biobert_v1.1_pubmed/vocab.txt')
tokenizer = BertTokenizer('disease_name_recog/biobert_v1.1_pubmed/vocab.txt')


def fetch_sentences(file_path):
    """
    :param file_path: file.tsv
    :return: sentences -> a list of lists of token strings
             tags      -> a list of lists of tag strings
    """
    sentences = []
    tags = []
    sent = []
    tag = []
    with open(file_path) as tsv_f:
        reader = csv.reader(tsv_f, delimiter='\t')
        for row in reader:
            if len(row) == 0:
                if len(sent) != len(tag):
                    break
                sentences.append(sent)
                tags.append(tag)
                sent = []
                tag = []
            else:
                sent.append(row[0])
                tag.append(row[1])
    return sentences, tags


def get_inputs_labels(sentences, tags, max_length):
    labels = []
    input_ids = []
    real_sentence_length = []
    for sentence, tag in zip(sentences, tags):
        token_sentence = []
        labels_per_sentence = []
        for word, label in zip(sentence, tag):
            token_word = tokenizer.tokenize(word)
            nb_tokens = len(token_word)
            token_sentence.extend(token_word)
            labels_per_sentence.extend([label] * nb_tokens)

        real_sentence_length.append(len(token_sentence))
        input_id = pad_sequences([tokenizer.convert_tokens_to_ids(token_sentence)],
                                 maxlen=max_length, dtype="long", value=0.0,
                                 truncating="post", padding="post")

        input_ids.extend(input_id)
        labels.append(labels_per_sentence)

    attention_masks = [[float(i != 0.0) for i in input_id] for input_id in input_ids]
    unique_tags = list(set(tag for doc in tags for tag in doc))
    unique_tags.sort()
    unique_tags.append("PAD")
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    encoded_labels = pad_sequences([[tag2id[tag] for tag in doc] for doc in labels],
                                   maxlen=max_length, dtype="long", value=tag2id["PAD"],
                                   truncating="post", padding="post")

    return np.array(input_ids), np.array(attention_masks), encoded_labels, real_sentence_length
