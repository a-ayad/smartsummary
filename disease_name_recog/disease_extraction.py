import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.dataset_preprocessing import fetch_sentences
from model.biobert_ner import sparse_categorical_accuracy_masked, sparse_crossentropy_masked
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)
tokenizer = BertTokenizer('biobert_v1.1_pubmed/vocab.txt')
notes_df = pd.read_csv("data/Notes_10patients.csv", error_bad_lines=False, skip_blank_lines=False)
MAX_LENGTH = 125

subject_id_df = notes_df["'subject_id'"]
patient_id_df = subject_id_df[subject_id_df.str.isnumeric() == True]
# print(patient_id)
# print(patient_id.index)
unique_patient_df = patient_id_df[:10]

patient_ids_index = unique_patient_df.index.to_list()
patient_ids = unique_patient_df.to_list()[:9]
keywords = ['service:', 'chief complaint', 'history of present illness', 'past medical history']

info = notes_df["'row_id'"]
patient_info = {}


def extract_info(keyword, info_df):
    info_sentences = ''
    keyword_info_df = info_df[info_df.str.lower().str.contains(keyword, na=False)]
    if len(keyword_info_df) == 0:
        info_sentences += 'nan'
    else:
        for idx in range(100):
            info_element = info.iloc[keyword_info_df.index+idx].item()
            if str(info_element) == 'nan':
                break
            else:
                info_sentences += (info_element + " ")
    return info_sentences


for id_1, id_2, patient in zip(patient_ids_index[:-1], patient_ids_index[1:], patient_ids):
    patient_info_df = info[id_1:id_2]
    patient_info[patient] = {}
    for keyword in keywords:
        info_sentences = extract_info(keyword, patient_info_df)
        patient_info[patient][keyword] = info_sentences

print(patient_info)


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

actual_sentences = []
pred_labels = []

patient_disease = {}
for patient, keys in patient_info.items():
    patient_disease[patient] = {}
    for key, text in keys.items():
        print(text)
        input_id, mask = get_inputs([text], MAX_LENGTH)
        y_pred = biobert_model.predict([input_id, mask])
        pred_tags = np.argmax(y_pred, 2)[0]

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
        patient_disease[patient][key] = diseases

diseases = []
for i in range(len(actual_sentences)):
    for idx, (a, p) in enumerate(zip(actual_sentences[i], pred_labels[i])):
        if p == 'B':
            disease = ""
            disease += a
            if idx == len(pred_labels[i]) - 1 or pred_labels[i][idx+1] != 'I':
                diseases.append([disease])
        elif p == 'I' and len(disease) != 0:
            disease += (" " + a)
            if idx == len(pred_labels[i]) - 1 or pred_labels[i][idx+1] == 'O':
                diseases.append([disease])
                disease = ""
print(diseases)
'''
for t, p in zip(actual_sentences, pred_labels):
    print(t)
    print('next')
    print(p)
'''

print(actual_sentences[0])
print(pred_labels[0])
