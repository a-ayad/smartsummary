import pandas as pd
import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from disease_name_recog.data.dataset_preprocessing import fetch_sentences
from disease_name_recog.model.biobert_ner import sparse_categorical_accuracy_masked, sparse_crossentropy_masked
from dataloaders.recsys_data_loader import RecSys_DataLoader
import icd_coding.models as md
from dataloaders.icd_coding_data_loader import load_vocab
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

FILEPATH = 'dataloaders/data/recsys_data/full_dataV3.csv'
VOCAB_FILE = "dataloaders/data/icd_coding_data/vocab.csv"
RECSYS_MODEL_PATH = 'recsys/recsys_model'
MAX_LENGTH = 125
tokenizer = BertTokenizer('disease_name_recog/biobert_v1.1_pubmed/vocab.txt')
biobert_model = tf.keras.models.load_model('disease_name_recog/model/biobert_ner',
                                               custom_objects={
                                                   'sparse_categorical_accuracy_masked': sparse_categorical_accuracy_masked,
                                                   'sparse_crossentropy_masked': sparse_crossentropy_masked})

file = 'dataloaders/data/disease_name_recog/train_dev.tsv'
sentences, tags = fetch_sentences(file)
unique_tags = list(set(tag for doc in tags for tag in doc))
unique_tags.sort()
unique_tags.append("PAD")
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
# tag2id = {'B': 0, 'O': 1, 'I': 2, 'PAD': 3}
id2tag = {id: tag for tag, id in tag2id.items()}


def vector2code(vector, all_codes):
    # t = [all_codes[i + 1] for i, prob in enumerate(pr[0]) if prob > 0.5]
    codes = [all_codes[i] for i, code in enumerate(vector) if int(code) == 1]
    return codes


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

    recsys_dataloader = RecSys_DataLoader(FILEPATH)
    data_df = recsys_dataloader.get_data_df()
    vocab = load_vocab(VOCAB_FILE)
    top_50_codes = load_vocab("dataloaders/data/icd_coding_data/TOP_50_CODES.csv")

    icd_coding_model = md.CNN(vocab=vocab)
    icd_coding_model = icd_coding_model.build_model(50)
    icd_coding_model.load_weights("icd_coding/training/CNN_Nov30/best_model")

    # print(biobert_model.summary())


    try:
        while True:
            # 195044
            # 142745
            # 161707
            # 186679
            hadm_id = int(input())
            user_notes = data_df[data_df['HADM_ID'] == hadm_id]['TEXT'].values
            user_age = data_df[data_df['HADM_ID'] == hadm_id]['age'].values
            user_weight = data_df[data_df['HADM_ID'] == hadm_id]['Weight_kg'].values
            icustay_id = data_df[data_df['HADM_ID'] == hadm_id]['icustayid'].item()

            user_icd_codes_binary = np.asarray(icd_coding_model.predict(user_notes)).round()
            user_icd_codes = vector2code(user_icd_codes_binary[0], top_50_codes)

            user_disease_name = str(list(set(get_disease(user_notes[0])[0])))

            user_profile_dict = {"hadm_id": hadm_id,
                                 "codes": user_icd_codes_binary,
                                 "age": user_age,
                                 "weight": user_weight,
                                 "disease": user_disease_name}
            user_profile = (user_profile_dict['age'],
                            user_profile_dict['weight'],
                            user_profile_dict['codes'],
                            np.expand_dims(user_profile_dict['disease'], axis=0))

            recsys = tf.keras.models.load_model(RECSYS_MODEL_PATH)
            lab_ratings_pred = recsys.predict(user_profile)[0]
            top5_lab = np.argsort(lab_ratings_pred)[:5]

            lab_values_df = pd.read_csv("dataloaders/data/tsc_data/stacked_input_5.csv")
            LAB_NAMES = lab_values_df.columns.to_list()[17:-1]
            user_lab_values = lab_values_df[lab_values_df['icustayid'] == icustay_id].iloc[-5:].iloc[:,
                              17:-1].to_numpy()
            lab_pair = list(zip(LAB_NAMES, user_lab_values.T))
            lab_dict_key = [x for x in range(len(lab_pair))]
            lab_dict = dict(zip(lab_dict_key, lab_pair))
            res = [lab_dict[x] for x in top5_lab]

            user_profile_dict['codes'] = user_icd_codes
            print(f"User Profile: {user_profile_dict}")
            print(f"Top 5 recommended lab values (previous 5 timestamps): {res}")
            print(f"All lab values: {lab_dict}")
    except KeyboardInterrupt:
        pass

    print('Closed')
    user_next_lab_values = tsc(user_lab_values)

