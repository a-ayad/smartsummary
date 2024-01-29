import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_addons as tfa
from tensorflow.python.keras import backend as K
from tensorflow.keras import initializers
import pandas as pd


class CNN_Attention(tf.keras.layers.Layer):

    def __init__(self, class_num, **kwargs):
        super(CNN_Attention, self).__init__(class_num)
        self.class_num = class_num
        self.filter_num = None
        self.Wa = None

    def build(self, input_shape):
        self.filter_num = input_shape[2]

        # self.Wa = (the number of classes, the number of filters)
        self.Wa = self.add_weight(shape=(self.class_num, self.filter_num),
                                  initializer=initializers.get('glorot_uniform'),
                                  trainable=True,
                                  name='weights')

        super(CNN_Attention, self).build(input_shape)

    def call(self, inputs):
        # inputs_trans = (batch_size, the number of filters, sentence_length)
        inputs_trans = tf.transpose(inputs, [0, 2, 1])

        # at = (batch_size, the number of classes, sentence_length)
        at = tf.matmul(self.Wa, inputs_trans)

        # Softmax
        at = K.exp(at - K.max(at, axis=-1, keepdims=True))
        attention_weights = at / K.sum(at, axis=-1, keepdims=True)
        # print(attention_weights)
        # weighted sum
        # v = (batch_size, the number of classes, the number of filters)
        attention_adjusted_output = K.batch_dot(attention_weights, inputs)

        return attention_weights, attention_adjusted_output

    def get_config(self):
        config = super(CNN_Attention, self).get_config()
        config['class_num'] = self.class_num
        config['filter_num'] = None
        config['Wa'] = None
        return config


CNN_FILTERS = 500
CNN_KERNEL_SIZE = 4
EMBEDDING_DIM = 100
DROPOUT = 0.2
LEARNING_RATE = 0.003
BATCH_SIZE = 32
MAX_LENGTH = 2500
EPOCHS = 1


class CNNAttention:

    def __init__(self, vocab, args=None, verbose=True):
        self.args = vars(args) if args is not None else {}
        self.model = None
        self.verbose = verbose
        output_length = self.args.get("max_length")
        self.vectorize_layer_voc = TextVectorization(standardize="lower_and_strip_punctuation",
                                                     max_tokens=len(vocab)+2,
                                                     output_mode='int',
                                                     vocabulary=vocab,
                                                     output_sequence_length=output_length)

    def build_model(self, output_shape):
        filters = self.args.get("cnn_filters", CNN_FILTERS)
        kernel_size = self.args.get("cnn_kernel_size", CNN_KERNEL_SIZE)
        embedding_dim = self.args.get("embedding_dim", EMBEDDING_DIM)
        learning_rate = self.args.get("learning_rate", LEARNING_RATE)
        dropout = self.args.get("dropout", DROPOUT)

        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = self.vectorize_layer_voc(text_input)
        x = keras.layers.Embedding(len(self.vectorize_layer_voc.get_vocabulary()), embedding_dim, mask_zero=True)(x)
        x = keras.layers.Dropout(dropout)(x)
        cnn1 = keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        att_weights, att1 = CNN_Attention(output_shape)(cnn1)
        att1 = keras.layers.Dropout(dropout)(att1)
        label_li = []

        for i in range(output_shape):
            label_li.append(keras.layers.Dense(1, activation='sigmoid')(att1[::, i]))

        labels_li = tf.stack(label_li, axis=1)
        labels_li = tf.squeeze(labels_li, [2])

        model = tf.keras.Model(inputs=text_input, outputs=labels_li)

        f1_macro = tfa.metrics.F1Score(num_classes=output_shape, threshold=0.5, average='macro', name='f1_score_macro')
        f1_micro = tfa.metrics.F1Score(num_classes=output_shape, threshold=0.5, average='micro', name='f1_score_micro')
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[keras.metrics.AUC(multi_label=True),
                               keras.metrics.Recall(),
                               keras.metrics.Precision(),
                               f1_macro,
                               f1_micro])
        return model

    def fit(self, X, y, validation_data=None, callbacks=None):
        if not self.model:
            self.model = self.build_model(y[0].shape[0])

        if self.verbose:
            self.model.summary()

        batch_size = self.args.get("batch_size", BATCH_SIZE)
        epochs = self.args.get("epochs", EPOCHS)
        self.model.fit(X, y,
                       validation_data=validation_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks,
                       shuffle=True)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save(path)


if __name__ == '__main__':
    voc_df = pd.read_csv("mimicdata/vocab.csv", header=None)
    vocab = sorted(voc_df.iloc[:, 0].tolist())
    output_length = 2500
    # data = [
    #     "The sky is blue.",
    #     "Grass is green.",
    #     "Hunter2 is my password.",
    # ]
    #
    # # Create vectorizer.
    # text_dataset = tf.data.Dataset.from_tensor_slices(data)
    # vectorizer = TextVectorization(
    #     max_tokens=100000, output_mode='tf-idf', ngrams=None,
    # )
    # vectorizer.adapt(text_dataset.batch(1024))
    text_dataset = tf.data.Dataset.from_tensor_slices(vocab)
    vectorize_layer_voc = TextVectorization(standardize="lower_and_strip_punctuation",
                                            max_tokens=len(vocab) + 2,
                                            output_mode='int',
                                            output_sequence_length=output_length)
    vectorize_layer_voc.adapt(text_dataset.batch(64))
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
    x = vectorize_layer_voc(text_input)
    x = keras.layers.Embedding(len(vectorize_layer_voc.get_vocabulary()), 100)(x)
    x = keras.layers.Dropout(0.2)(x)
    cnn1 = keras.layers.Conv1D(500, 4, padding='same', activation='relu')(x)

    att_weights, att1 = CNN_Attention(50)(cnn1)
    att1 = keras.layers.Dropout(0.2)(att1)

    label_li = []

    for i in range(50):
        label_li.append(keras.layers.Dense(1, activation='sigmoid')(att1[::, i]))

    labels_li = tf.stack(label_li, axis=1)
    labels_li = tf.squeeze(labels_li, [2])


    model = tf.keras.Model(inputs=text_input, outputs=labels_li)
    f1_macro = tfa.metrics.F1Score(num_classes=50, threshold=0.5, average='macro', name='f1_score_macro')
    f1_micro = tfa.metrics.F1Score(num_classes=50, threshold=0.5, average='micro', name='f1_score_micro')
    # model.compile(loss='binary_crossentropy',
    #              optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #              metrics=[keras.metrics.AUC(multi_label=True),
    #                       keras.metrics.Recall(),
    #                       keras.metrics.Precision(),
    #                       f1_macro,
    #                       f1_micro])
    # Save.
    filepath = "tmp-model"
    model.save(filepath, save_format="tf")

    # Load.
    loaded_model = tf.keras.models.load_model(filepath)
    print('test')


