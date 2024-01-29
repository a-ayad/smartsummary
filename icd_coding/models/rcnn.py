import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow_addons as tfa

from icd_coding.base.base_model import BaseModel
from icd_coding.attention.attention_layer import Attention

LSTM_DIM = 128
CNN_FILTERS = 500
CNN_KERNEL_SIZE = 4
EMBEDDING_DIM = 100
DROPOUT = 0.2
LEARNING_RATE = 0.003
BATCH_SIZE = 32
MAX_LENGTH = 2500
EPOCHS = 100


class RCNN(BaseModel):

    def __init__(self, vocab, args=None, verbose=True):
        self.args = vars(args) if args is not None else {}
        self.model = None
        self.verbose = verbose
        output_length = self.args.get("max_length")
        self.vectorize_layer_voc = TextVectorization(standardize="lower_and_strip_punctuation",
                                                     max_tokens=len(vocab) + 2,
                                                     output_mode='int',
                                                     output_sequence_length=output_length)
        text_dataset = tf.data.Dataset.from_tensor_slices(vocab)
        self.vectorize_layer_voc.adapt(text_dataset.batch(64))

    def build_model(self, output_shape):
        filters = self.args.get("cnn_filters", CNN_FILTERS)
        kernel_size = self.args.get("cnn_kernel_size", CNN_KERNEL_SIZE)
        embedding_dim = self.args.get("embedding_dim", EMBEDDING_DIM)
        dropout = self.args.get("dropout", DROPOUT)
        learning_rate = self.args.get("learning_rate", LEARNING_RATE)

        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = self.vectorize_layer_voc(text_input)
        x = keras.layers.Embedding(len(self.vectorize_layer_voc.get_vocabulary()), embedding_dim, mask_zero=True)(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
        conv = keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu')(x)

        if self.args.get("use_attention"):
            att_weights, att = Attention(output_shape, name='attention')(conv)
            att = keras.layers.Dropout(dropout)(att)
            predictions = []

            for i in range(output_shape):
                predictions.append(keras.layers.Dense(1, activation='sigmoid')(att[::, i]))

            predictions = tf.stack(predictions, axis=1)
            predictions = tf.squeeze(predictions, [2])
        else:
            conv = keras.layers.GlobalMaxPooling1D()(conv)
            conv = keras.layers.Dropout(dropout)(conv)
            predictions = keras.layers.Dense(output_shape, activation='sigmoid', name='predictions')(conv)

        model = tf.keras.Model(text_input, predictions)
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


    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--cnn_filters", type=int, default=CNN_FILTERS)
        parser.add_argument("--cnn_kernel_size", type=int, default=CNN_KERNEL_SIZE)
        parser.add_argument("--lstm_dim", type=int, default=LSTM_DIM)
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
        parser.add_argument("--epochs", type=int, default=EPOCHS)
        parser.add_argument("--use_attention", action="store_true", default=False)
        return parser