import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.dataset_preprocessing import fetch_sentences
from model.biobert_ner import sparse_categorical_accuracy_masked, sparse_crossentropy_masked
import numpy as np
import pandas as pd

np.random.seed(42)
tf.random.set_seed(42)
tokenizer = BertTokenizer('biobert_v1.1_pubmed/vocab.txt')
MAX_LENGTH = 125
biobert_model = tf.keras.models.load_model('model/biobert_ner',
                                           custom_objects={'sparse_categorical_accuracy_masked': sparse_categorical_accuracy_masked,
                                                           'sparse_crossentropy_masked': sparse_crossentropy_masked})
print(biobert_model.summary())
file = 'data/train_dev.tsv'
sentences, tags = fetch_sentences(file)
unique_tags = list(set(tag for doc in tags for tag in doc))
unique_tags.sort()
unique_tags.append("PAD")
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
# tag2id = {'B': 0, 'O': 1, 'I': 2, 'PAD': 3}
id2tag = {id: tag for tag, id in tag2id.items()}


def get_inputs(sentences, max_length):
    input_ids = []
    for sentence in sentences:
        token_sentence = []
        for idx, word in enumerate(sentence.split()):
            token_word = tokenizer.tokenize(word)
            nb_tokens = len(token_word)
            token_sentence.extend(token_word)

        input_id = pad_sequences([tokenizer.convert_tokens_to_ids(token_sentence)],
                                 maxlen=max_length, dtype="long", value=0.0,
                                 truncating="post", padding="post")

        input_ids.extend(input_id)

    attention_masks = [[float(i != 0.0) for i in input_id] for input_id in input_ids]

    return np.array(input_ids), np.array(attention_masks)
'''
text = "liver lacerations"
input_id, mask = get_inputs([text], MAX_LENGTH)
y_pred = biobert_model.predict([input_id, mask])
pred_tags = np.argmax(y_pred, 2)[0]

actual_sentences = []
pred_labels = []
token_sentence = []
for idx, word in enumerate(text.split()):
    # print(word)
    token_word = tokenizer.tokenize(word)
    # print(token_word)
    token_sentence.extend(token_word)

new_tokens, new_labels = [], []
for token, label_idx in zip(token_sentence, pred_tags):
    # print(token)
    # print(label_idx)
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(id2tag[label_idx])
        new_tokens.append(token)
actual_sentences.append(new_tokens)
pred_labels.append(new_labels)
# print(key)
# print(new_tokens)
# print(new_labels)
diseases = []
for idx, (a, p) in enumerate(zip(new_tokens, new_labels)):
    if p == 'B':
        disease = ""
        disease += a
        if idx == len(new_labels) - 1 or new_labels[idx + 1] != 'I':
            diseases.append([disease])
    elif p == 'I' and len(disease) != 0:
        disease += (" " + a)
        if idx == len(new_labels) - 1 or new_labels[idx + 1] == 'O':
            diseases.append([disease])
            disease = ""

print(diseases)
'''


def get_disease(text):
    input_id, mask = get_inputs([text], MAX_LENGTH)
    y_pred = biobert_model.predict([input_id, mask])
    pred_tags = np.argmax(y_pred, 2)[0]

    actual_sentences = []
    pred_labels = []
    token_sentence = []
    for idx, word in enumerate(text.split()):
        # print(word)
        token_word = tokenizer.tokenize(word)
        # print(token_word)
        token_sentence.extend(token_word)

    new_tokens, new_labels = [], []
    for token, label_idx in zip(token_sentence, pred_tags):
        # print(token)
        # print(label_idx)
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(id2tag[label_idx])
            new_tokens.append(token)
    actual_sentences.append(new_tokens)
    pred_labels.append(new_labels)
    # print(key)
    # print(new_tokens)
    # print(new_labels)
    diseases = []
    """
    for idx, (a, p) in enumerate(zip(new_tokens, new_labels)):
        if p == 'B':
            disease = ""
            disease += a
            if idx == len(new_labels) - 1 or new_labels[idx + 1] != 'I':
                diseases.append([disease])
        elif p == 'I' and not disease:
            continue
        elif p == 'I' and len(disease) != 0:
            disease += (" " + a)
            if idx == len(new_labels) - 1 or new_labels[idx + 1] == 'O':
                diseases.append([disease])
                disease = ""
        """
    idx = 0
    while idx < len(new_tokens):
        if new_labels[idx] == 'B':
            disease = ""
            disease += new_tokens[idx]
            # last token
            # if idx == len(new_labels)-1:
            #     break
            # else:
            for i in range(idx+1, idx+50):
                # last token
                if i == len(new_tokens)-1:
                    if disease:
                        diseases.append(disease)
                    idx = i
                    break
                elif new_labels[i] == 'I':
                    disease += (" " + new_tokens[i])
                else:
                    diseases.append(disease)
                    idx = i
                    break
        else:
            idx += 1

    return diseases, actual_sentences, pred_labels


if __name__ == '__main__':
    data_dev_df = pd.read_csv("dev_50.csv")
    text_dev = data_dev_df["TEXT"].to_numpy()
    for i in range(10):
        print(get_disease(text_dev[i]))
    print('done')
    input()