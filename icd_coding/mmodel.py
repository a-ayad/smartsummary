
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from icd_coding.models import CNN
from data_loader.mimic_data_loader import MimicDataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.random.set_seed(1)

tokenizer = RegexpTokenizer(r'\w+')


@tf.function
def as_labels(x):
    mask = tf.dtypes.cast(x, tf.bool)
    s = tf.shape(mask)
    r = tf.reshape(tf.range(s[-1]) + 1, tf.concat([tf.ones(tf.rank(x) - 1, tf.int32), [-1]], axis=0))
    r = tf.tile(r, tf.concat([s[:-1], [1]], axis=0))
    return tf.ragged.boolean_mask(r, mask)


def convert2np(t):
    return t.numpy()


def clean(note):
    tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric()]
    text = '' + ' '.join(tokens) + ''
    return text


def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for i, line in enumerate(ef):
            if i == 1:
                print("Adding UNK embedding")
                vec = np.random.randn(len(W[-1]))
                vec = vec / float(np.linalg.norm(vec) + 1e-6)
                W.append(vec)
                print("UNK embedding added")

            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
    W = np.array(W)
    return W


def clean_icd_labels(icd_label):
    res = icd_label.replace("[", "")
    res = res.replace("]", "")
    res = res.strip()
    return res


def sp(text):
    return text.split()


def toarr(label):
    return np.fromstring(label, sep=' ').astype('float32')


data = pd.read_csv('full_dataV3.csv')
data = data.drop(data[data['disease'].map(len) < 3].index)
data = data.drop(data[data["icd_codes_labels"] != data["icd_codes_labels"]].index)

lab_ratings = data.iloc[:, 2:-7].to_numpy()
# data['TEXT'] = data['TEXT'].apply(clean)

text = data['TEXT'].to_numpy()
hadm_id = data['HADM_ID'].to_numpy(dtype=str)
unique_hadm_id = np.unique(hadm_id)

data_dir = "mimicdata/"
dataloader = MimicDataLoader(data_dir)
vocab = dataloader.get_vocab()
'''
CNN = CNN(vocab=vocab)
icd_model = CNN.build_model(50)
icd_model.load_weights('training/CNN_Test/best_model')
icd_codes = (np.asarray(icd_model.predict(text))).round()
print(icd_codes)

icd_codes = tf.keras.preprocessing.sequence.pad_sequences(as_labels(icd_codes).numpy(), padding='post')
'''

user_age = data['age'].to_numpy()
user_weight_kg = data['Weight_kg'].to_numpy()
disease = data['disease'].to_numpy()
icd_codes = np.asarray([toarr(x) for x in data['icd_codes_labels']])
icd_codes = tf.keras.preprocessing.sequence.pad_sequences(icd_codes, padding='post')
data_dic = {"hadm_id": hadm_id,
            "codes": icd_codes,
            "age": user_age,
            "weight": user_weight_kg,
            "disease": disease}


class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, agg_mode='sum', **kwargs):
        super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)

        if agg_mode not in ['sum', 'mean']:
            raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
        self.agg_mode = agg_mode

    @tf.function
    def call(self, inputs, mask=None):
        masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
        if self.agg_mode == 'sum':
            aggregated = tf.reduce_sum(masked_embeddings, axis=1)
        elif self.agg_mode == 'mean':
            aggregated = tf.reduce_mean(masked_embeddings, axis=1)

        return aggregated

    def get_config(self):
        # this is used when loading a saved model that uses a custom layer
        return {'agg_mode': self.agg_mode}


class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        embedding_dimension = 64
        vectorize_layer = layers.experimental.preprocessing.TextVectorization(
            max_tokens=len(vocab) + 2,
            output_mode='int',
            vocabulary=vocab
        )
        embedding_matrix = load_embeddings("/Users/dai/Desktop/caml-mimic/mimicdata/mimic3/processed_full.embed")


        self.icd_embedding = tf.keras.Sequential(
            [
                layers.Embedding(51, embedding_dimension, mask_zero=True),
                # (?, input_length, embedding_dimension)
                # layers.Flatten(),
                layers.GlobalAveragePooling1D()
                # (?, input_length x embedding_dimension)
                # L2NormLayer(),
                # MaskedEmbeddingsAggregatorLayer(agg_mode='mean')
            ]
        )


        # self.user_embeddings = tf.keras.Sequential(
        #     [
        #         layers.experimental.preprocessing.StringLookup(vocabulary=unique_hadm_id, mask_token=None),
        #         layers.Embedding(len(unique_hadm_id) + 1, embedding_dimension)
        #         # (?, 1, embed_dim)
        #     ]
        # )

        self.normalized_age_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
        self.normalized_age_layer.adapt(user_age)

        self.normalized_weight_layer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
        self.normalized_weight_layer.adapt(user_weight_kg)


        # self.text_embeddings = tf.keras.Sequential(
        #     [
        #         text_vectorizer,
        #         layers.Embedding(sss, embedding_dimension)

        self.disease_embeddings = tf.keras.Sequential(
            [
                vectorize_layer,
                layers.Embedding(
                    len(vocab)+2,
                    100,
                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                    trainable=False,
                    mask_zero=True),
                # layers.Flatten(),
                # L2NormLayer(),
                # MaskedEmbeddingsAggregatorLayer(agg_mode='mean')
                layers.GlobalAveragePooling1D()
             ]
        )

        self.rating_model = tf.keras.Sequential(
            [
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(25)
            ]
        )

    def call(self, inputs):
        # id = inputs['hadm_id']
        code = inputs['codes']
        disease_name = inputs['disease']

        # user_embedd = self.user_embeddings(id)
        # user_embedd = tf.squeeze(user_embedd)

        icd_codes_embedding = self.icd_embedding(code)
        disease_name_embedding = self.disease_embeddings(disease_name)

        normalized_age = self.normalized_age_layer(inputs["age"])
        normalized_weight = self.normalized_weight_layer(inputs["weight"])

        user_feat = tf.concat([
            # user_embedd,
            icd_codes_embedding,
            normalized_age,
            normalized_weight,
            disease_name_embedding
        ], axis=1)
        return self.rating_model(user_feat)


model = MyModel()
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    clipnorm=1.0
)
model.compile('rmsprop', 'mse', run_eagerly=True)
# model.fit()
model.fit(data_dic, lab_ratings, batch_size=64, epochs=1)
print(model.summary())
model.compile('rmsprop', 'mse', run_eagerly=False)
model.fit(data_dic, lab_ratings, batch_size=64, epochs=50)
pred_b4_fb = model.predict(data_dic)

ic_embed = model.icd_embedding
dis_embed = model.disease_embeddings

fb_icd_test = icd_codes
fb_dis_test = disease
fb_age_test = user_age
fb_weight_test = user_weight_kg

ic_embed_mean = ic_embed.predict(fb_icd_test).mean(axis=-1)
dis_embed_mean = dis_embed.predict(fb_dis_test).mean(axis=-1)

fb_df = data.copy()

fb_df['ic_embed_mean'] = ic_embed_mean
fb_df['dis_embed_mean'] = dis_embed_mean

fb_dataset = np.stack((fb_age_test, fb_weight_test, dis_embed_mean, ic_embed_mean), axis=1)
scaler = StandardScaler()
scaled_fb_dataset = scaler.fit_transform(fb_dataset)
# pca = PCA(n_components=2)
# fb_pca = pca.fit(scaled_fb_dataset).transform(scaled_fb_dataset)
k = 25
kmeans = KMeans(n_clusters=k)
cluster_labels = kmeans.fit_predict(scaled_fb_dataset)
fb_df['cluster_labels'] = cluster_labels


clusters_fb_ratings = []
clusters_fb_user_age = []
clusters_fb_user_weight = []
clusters_fb_diseases = []
clusters_fb_icd_codes = []

for i in range(k):
    clusters_fb_ratings.append(fb_df[fb_df['cluster_labels'] == i].iloc[:, 2:-10])
    clusters_fb_diseases.append(fb_df[fb_df['cluster_labels'] == i]['disease'].to_numpy())
    clusters_fb_user_age.append(fb_df[fb_df['cluster_labels'] == i]['age'].to_numpy())
    clusters_fb_user_weight.append(fb_df[fb_df['cluster_labels'] == i]['Weight_kg'].to_numpy())

    icd = fb_df[fb_df['cluster_labels'] == i]['icd_codes_labels']
    icd_codes_fb = np.asarray([toarr(x) for x in icd])
    icd_codes_fb = tf.keras.preprocessing.sequence.pad_sequences(icd_codes_fb, padding='post', maxlen=25)
    clusters_fb_icd_codes.append(icd_codes_fb)


for c in clusters_fb_ratings:
    for i in range(0, 25):
        c.iloc[:, i] = np.random.randint(1, 11)


fb_train_data = []
fb_test_data = []
fb_train_labels = []
fb_test_labels = []


for i in range(k):
    fb_age_train, fb_age_test, fb_weight_train, fb_weight_test = train_test_split(
        clusters_fb_user_age[i],
        clusters_fb_user_weight[i],
        test_size=0.2,
        random_state=42)

    fb_diseases_train, fb_diseases_test, fb_icd_codes_train, fb_icd_codes_test = train_test_split(
        clusters_fb_diseases[i],
        clusters_fb_icd_codes[i],
        test_size=0.2,
        random_state=42)

    fb_ratings_train, fb_ratings_test, _, _ = train_test_split(
        clusters_fb_ratings[i].to_numpy(),
        clusters_fb_icd_codes[i],
        test_size=0.2,
        random_state=42)
    if i == 0:
        fb_icd_codes_train_data = fb_icd_codes_train
        fb_age_train_data = fb_age_train
        fb_weight_train_data = fb_weight_train
        fb_diseases_train_data = fb_diseases_train
        fb_icd_codes_test_data = fb_icd_codes_test
        fb_age_test_data = fb_age_test
        fb_weight_test_data = fb_weight_test
        fb_diseases_test_data = fb_diseases_test
        fb_labels_train = fb_ratings_train
        fb_labels_test = fb_ratings_test
    else:
        fb_icd_codes_train_data = np.vstack((fb_icd_codes_train_data, fb_icd_codes_train))
        fb_age_train_data = np.hstack((fb_age_train_data, fb_age_train))
        fb_weight_train_data = np.hstack((fb_weight_train_data, fb_weight_train))
        fb_diseases_train_data = np.hstack((fb_diseases_train_data, fb_diseases_train))

        fb_icd_codes_test_data = np.vstack((fb_icd_codes_test_data, fb_icd_codes_test))
        fb_age_test_data = np.hstack((fb_age_test_data, fb_age_test))
        fb_weight_test_data = np.hstack((fb_weight_test_data, fb_weight_test))
        fb_diseases_test_data = np.hstack((fb_diseases_test_data, fb_diseases_test))

        fb_labels_train = np.vstack((fb_labels_train, fb_ratings_train))
        fb_labels_test = np.vstack((fb_labels_test, fb_ratings_test))

    fb_data_dict_train = {
        "codes": fb_icd_codes_train_data,
        "age": fb_age_train_data,
        "weight": fb_weight_train_data,
        "disease": fb_diseases_train_data
    }

    fb_data_dict_test = {
        "codes": fb_icd_codes_test_data,
        "age": fb_age_test_data,
        "weight": fb_weight_test_data,
        "disease": fb_diseases_test_data
    }

    fb_data_dict_val = {
        "codes": fb_icd_codes_test,
        "age": fb_age_test,
        "weight": fb_weight_test,
        "disease": fb_diseases_test
    }

    fb_train_data.append(fb_data_dict_train)
    fb_test_data.append(fb_data_dict_val)

    fb_train_labels.append(fb_ratings_train)
    fb_test_labels.append(fb_ratings_test)

# fb_data = pd.read_csv('feedback.csv')
# fb_ratings = fb_data.iloc[:, 2:-4].to_numpy()
# fb_data['TEXT'] = fb_data['TEXT'].apply(clean)
# fb_text = fb_data['TEXT'].to_numpy()
# fb_hadm_id = fb_data['HADM_ID'].to_numpy(dtype=str)
# fb_icd_codes = (np.asarray(icd_model.predict(fb_text))).round()
# fb_user_age = fb_data['age'].to_numpy()
# fb_user_weight_kg = fb_data['Weight_kg'].to_numpy()
# fb_data_dic = {"hadm_id": fb_hadm_id,
#                "codes": fb_icd_codes,
#                "age": fb_user_age,
#                "weight": fb_user_age}
# model.icd_embedding.layers[0].trainable = False
model.disease_embeddings.layers[1].trainable = False
# # model_fb = MyModel()
# model.compile('rmsprop', 'mse', run_eagerly=False)
top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
top5_acc.reset_state()
top5_acc.update_state(fb_labels_test, model.predict(fb_data_dict_test))
print(f"Before Feedback dataset Acc ALL: {top5_acc.result().numpy()}")

print(model.summary())
sample_w = np.full((fb_labels_train.shape[0], 1), 1.5)
model.fit(fb_data_dict_train, fb_labels_train, batch_size=64, epochs=100, sample_weight=sample_w,
          validation_data=(fb_data_dict_test, fb_labels_test))

unique, counts = np.unique(cluster_labels, return_counts=True)
print(dict(zip(unique, counts)))

top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
top5_acc.reset_state()
top5_acc.update_state(fb_labels_test, model.predict(fb_data_dict_test))
print(f"Acc ALL: {top5_acc.result().numpy()}")
'''
for i in range(k):
    sample_w = np.full((len(fb_train_labels[i]), 1), 1.5)
    print(f"K: {i}")
    print(f"training data size: {fb_train_labels[i].shape[0]}")
    model.fit(fb_train_data[i], fb_train_labels[i], batch_size=8, epochs=20, sample_weight=sample_w,
              validation_data=(fb_test_data[i], fb_test_labels[i]))
    for j in range(k):
        print(f"K: {j}")
        top5_acc.reset_state()
        top5_acc.update_state(fb_test_labels[j], model.predict(fb_test_data[j]))
        print(top5_acc.result().numpy())
'''

for i in range(k):
    top5_acc.reset_state()
    top5_acc.update_state(fb_test_labels[i], model.predict(fb_test_data[i]))
    print(f"Group {i}: {top5_acc.result().numpy()}")
    # print(top5_acc.result().numpy())

# pred_after_fb = model.predict(data_dic)
print("Top 5 Prediction sample 0-50 before training feedback dataset")
print(tf.math.top_k(pred_b4_fb[:50], 5)[1])
print("Top 5 Prediction sample 0-50 after training feedback dataset")
print(tf.math.top_k(pred_after_fb[:50], 5)[1])
print("Top 5 Prediction sample 1000-1050 before training feedback dataset")
print(tf.math.top_k(pred_b4_fb[1000:1050], 5)[1])
print("Top 5 Prediction sample 1000-1050 after training feedback dataset")
print(tf.math.top_k(pred_after_fb[1000:1050], 5)[1])
top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
top5_acc.update_state(fb_ratings, pred_after_fb[:500])
print(f"First 500 samples top 5 acc after training feedback dataset: {top5_acc.result().numpy()}")
top5_acc.reset_state()
top5_acc.update_state(lab_ratings[:500], pred_b4_fb[:500])
print(f"First 500 samples top 5 acc before training feedback dataset: {top5_acc.result().numpy()}")
top5_acc.reset_state()
top5_acc.update_state(lab_ratings[1000:1050], pred_after_fb[1000:1050])
print(f"Sample 1000 - 1050 top 5 acc after training feedback dataset: {top5_acc.result().numpy()}")
top5_acc.reset_state()
top5_acc.update_state(lab_ratings[1000:1050], pred_b4_fb[1000:1050])
print(f"Sample 1000 - 1050 top 5 acc before training feedback dataset: {top5_acc.result().numpy()}")



"""
data = pd.read_csv('recsys_data.csv')
feedback_data = data.iloc[:500, :].copy()
feedback_data.iloc[:, 2:-4] = np.random.randint(1, 11, feedback_data.iloc[:, 2:-4].shape)
feedback_data.to_csv('feedback.csv', index=False)
age = data['age'].to_numpy()
weight_kg = data['Weight_kg'].to_numpy()
demography = np.stack((age, weight_kg), axis=-1)
normalized_demography = tf.keras.layers.experimental.preprocessing.Normalization(
          axis=None
        )
normalized_demography.adapt(demography)

normalized_weight = tf.keras.layers.experimental.preprocessing.Normalization()
normalized_weight.adapt(weight_kg)

normalized_age = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
normalized_age.adapt(age)

input()
"""


def load_embeddings(embed_file):
    #also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for i, line in enumerate(ef):
            if i == 1:
                print("Adding UNK embedding")
                vec = np.random.randn(len(W[-1]))
                vec = vec / float(np.linalg.norm(vec) + 1e-6)
                W.append(vec)
                print("UNK embedding added")

            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
    W = np.array(W)
    return W

'''
vectorize_layer = layers.experimental.preprocessing.TextVectorization(
    max_tokens=len(vocab) + 2,
    output_mode='int',
    vocabulary=vocab
)

embedding_layer = layers.Embedding(
    len(vocab)+2,
    100,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True)
embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

input()
'''