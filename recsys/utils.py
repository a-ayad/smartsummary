import tensorflow as tf
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

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


def clustering(features, nb_clusters):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=nb_clusters)
    cluster_labels = kmeans.fit_predict(scaled_features)

    return cluster_labels


def create_fb_dataset(fb_df, nb_clusters, version='v2'):
    clusters_fb_ratings = []
    clusters_fb_user_age = []
    clusters_fb_user_weight = []
    clusters_fb_diseases = []
    clusters_fb_icd_codes = []

    for i in range(nb_clusters):
        clusters_fb_ratings.append(fb_df[fb_df['cluster_labels'] == i].iloc[:, 2:-9])
        clusters_fb_diseases.append(fb_df[fb_df['cluster_labels'] == i]['disease'].to_numpy())
        clusters_fb_user_age.append(fb_df[fb_df['cluster_labels'] == i]['age'].to_numpy())
        clusters_fb_user_weight.append(fb_df[fb_df['cluster_labels'] == i]['Weight_kg'].to_numpy())

        if version == 'v2':
            # v2
            icd = fb_df[fb_df['cluster_labels'] == i]['icd_codes']
            icd_codes_fb = np.asarray([toarr(clean_icd_labels(x)) for x in icd])
        else:
            # v1
            icd = fb_df[fb_df['cluster_labels'] == i]['icd_codes_labels']
            icd_codes_fb = np.asarray([toarr(clean_icd_labels(x)) for x in icd])
            icd_codes_fb = tf.keras.preprocessing.sequence.pad_sequences(icd_codes_fb, padding='post', maxlen=25)
        clusters_fb_icd_codes.append(icd_codes_fb)

    for c in clusters_fb_ratings:
        for i in range(0, 25):
            c.iloc[:, i] = np.random.randint(1, 11)

    fb_train_data = []
    fb_test_data = []
    fb_train_labels = []
    fb_test_labels = []

    for i in range(nb_clusters):

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

    return fb_data_dict_train, fb_labels_train, fb_data_dict_test, fb_labels_test, fb_test_data, fb_test_labels


def precision_at_k(y_true, y_pred, k):
    topk_true = np.flip(np.argsort(y_true), 1)[:, :k]
    topk_pred = np.flip(np.argsort(y_pred), 1)[:, :k]

    n_relevant = 0
    n_recommend = 0

    for t, p in zip(topk_true, topk_pred):
        # print(f"t:{t}")
        # print(f"p:{p}")
        n_relevant += len(np.intersect1d(t, p))
        # print(f"rev:{n_relevant}")
        n_recommend += len(p)
        # print(n_recommend)

    return float(n_relevant) / n_recommend

