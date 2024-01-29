import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from icd_coding.models import CNN
from data_loader.mimic_data_loader import MimicDataLoader
from nltk.tokenize import RegexpTokenizer

from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ner.data.dataset_preprocessing import fetch_sentences
from ner.model.biobert_ner import sparse_categorical_accuracy_masked, sparse_crossentropy_masked

from sklearn.metrics import top_k_accuracy_score
tokenizer = RegexpTokenizer(r'\w+')


def clean(note):
    note = note.replace("\\n", " ")
    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
    text = '' + ' '.join(tokens) + ''
    return text


data = pd.read_csv('icd_coding/recsys_data.csv')
lab_ratings = data.iloc[:, 2:-1].to_numpy()
data['TEXT'] = data['TEXT'].apply(clean)

"""
text = data['TEXT'].to_numpy()
hadm_id = data['HADM_ID'].to_numpy(dtype=str)
unique_hadm_id = np.unique(hadm_id)

data_dir = "icd_coding/mimicdata/"
dataloader = MimicDataLoader(data_dir)
vocab = dataloader.get_vocab()
CNN = CNN(vocab=vocab)
icd_model = CNN.build_model(50)
icd_model.load_weights('icd_coding/training/CNN_Test/best_model')
icd_codes = (np.asarray(icd_model.predict(text))).round()
print(icd_codes)
data_dic = {"hadm_id": hadm_id, "codes": icd_codes}
'''
embedding_dimension = 64

model = tf.keras.Sequential(
            [
                layers.Embedding(50, embedding_dimension, input_length=50),
                # (?, input_length, embedding_dimension)
                layers.Flatten(),
                # (?, input_length x embedding_dimension)
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(25)
            ]
        )

model.compile('rmsprop', 'mse')
# model.fit()
model.fit(icd_codes, lab_ratings, batch_size=64, epochs=50)
tes = model.predict(icd_codes)



# print(tf.math.top_k(tes, 5))
# print(tf.math.top_k(lab_values, 5))
# print(tes)


user_embeddings = tf.keras.Sequential(
    [
                layers.experimental.preprocessing.StringLookup(vocabulary=unique_hadm_id, mask_token=None),
                layers.Embedding(len(unique_hadm_id) + 1, embedding_dimension),
                # (10, 1, embed_dim)
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(25)
    ]
)
user_embeddings.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError())
user_embeddings.fit(hadm_id, lab_ratings, batch_size=64, epochs=50)
# user_embedd = user_embeddings.predict(hadm_id)
# squeeze

# self.title_text_embedding = tf.keras.Sequential([
#       tf.keras.layers.TextVectorization(max_tokens=max_tokens),
#       tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
#       # We average the embedding of individual words to get one embedding vector
#       # per title.
#       tf.keras.layers.GlobalAveragePooling1D(),

'''


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        embedding_dimension = 64
        self.icd_embedding = tf.keras.Sequential(
            [
                layers.Embedding(50, embedding_dimension, input_length=50),
                # (?, input_length, embedding_dimension)
                layers.Flatten(),
                # (?, input_length x embedding_dimension)
            ]
        )

        self.user_embeddings = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.StringLookup(vocabulary=unique_hadm_id, mask_token=None),
                layers.Embedding(len(unique_hadm_id) + 1, embedding_dimension)
                # (?, 1, embed_dim)
            ]
        )

        # self.text_embeddings = tf.keras.Sequential(
        #     [
        #         text_vectorizer,
        #         layers.Embedding(sss, embedding_dimension)
        #     ]
        # )

        self.rating_model = tf.keras.Sequential(
            [
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(25)
            ]
        )

    def call(self, inputs):
        id = inputs['hadm_id']
        code = inputs['codes']
        user_embedd = self.user_embeddings(id)
        user_embedd = tf.squeeze(user_embedd)
        icd_codes = self.icd_embedding(code)
        user_feat = tf.concat([user_embedd, icd_codes], axis=1)
        return self.rating_model(user_feat)


model = MyModel()
model.compile('rmsprop', 'mse', run_eagerly=True)
# model.fit()
model.fit(data_dic, lab_ratings, batch_size=64, epochs=50)

"""

tokenizer = BertTokenizer('disease_name_recog/biobert_v1.1_pubmed/vocab.txt')
MAX_LENGTH = 125
biobert_model = tf.keras.models.load_model('disease_name_recog/model/biobert_ner',
                                           custom_objects={'sparse_categorical_accuracy_masked': sparse_categorical_accuracy_masked,
                                                           'sparse_crossentropy_masked': sparse_crossentropy_masked})
print(biobert_model.summary())
file = 'disease_name_recog/data/train_dev.tsv'
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
            if idx == len(new_labels)-1:
                break
            else:
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

    # return diseases, actual_sentences, pred_labels
    return diseases
print('start')
# data_10 = pd.read_csv('icd_coding/data_10_patients.csv')
# data_10['TEXT'] = data_10['TEXT'].apply(clean)
data['disease'] = pd.Series([] * len(data))
for i in range(len(data)):
    data['disease'].iloc[i] = get_disease(data['TEXT'].iloc[i])
    print(i)
# data['disease'] = data['TEXT'].apply(get_disease)
data.to_csv('save_data.csv', index=False)
'''
get_disease(data['TEXT'].iloc[4])
for i in range(10):
    print(get_disease(text[i]))
print('done')
'''











