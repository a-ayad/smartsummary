import tensorflow as tf
from tensorflow.keras import layers
from recsys.utils import load_embeddings


EMBEDDING_FILE = "/Users/dai/Desktop/caml-mimic/mimicdata/mimic3/processed_full.embed"


class ModelV1(tf.keras.Model):
    def __init__(self, vocab, user_age, user_weight_kg, embedding_file=EMBEDDING_FILE):
        super(ModelV1, self).__init__()
        embedding_dimension = 64
        vectorize_layer = layers.experimental.preprocessing.TextVectorization(
            max_tokens=len(vocab) + 2,
            output_mode='int',
            vocabulary=vocab
        )
        embedding_matrix = load_embeddings(embedding_file)

        self.icd_embedding = tf.keras.Sequential(
            [
                layers.Embedding(51, embedding_dimension, mask_zero=True),
                # (?, input_length, embedding_dimension)
                # layers.Flatten(),
                layers.GlobalAveragePooling1D()
                # (?, input_length x embedding_dimension)
                # L2NormLayer(),
                # MaskedEmbeddingsAggregatorLayer(agg_mode='mean')
            ],
            name='ICD_EMBEDDING_MODEL'
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
             ],
            name='DISEASE_EMBEDDING_MODEL'
        )

        self.rating_model = tf.keras.Sequential(
            [
                layers.Dense(1024, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(25)
            ],
            name='RATING_MODEL'
        )

    def call(self, inputs):
        # id = inputs['hadm_id']
        # code = inputs['codes']
        # disease_name = inputs['disease']

        age = inputs[0]
        weight = inputs[1]
        code = inputs[2]
        disease = inputs[3]

        # user_embedd = self.user_embeddings(id)
        # user_embedd = tf.squeeze(user_embedd)

        icd_codes_embedding = self.icd_embedding(code)
        disease_name_embedding = self.disease_embeddings(disease)

        normalized_age = self.normalized_age_layer(age)
        normalized_weight = self.normalized_weight_layer(weight)

        user_feat = tf.concat([
            # user_embedd,
            icd_codes_embedding,
            normalized_age,
            normalized_weight,
            disease_name_embedding
        ], axis=1)
        return self.rating_model(user_feat)

    @staticmethod
    def add_to_argparse(parser):
        return parser