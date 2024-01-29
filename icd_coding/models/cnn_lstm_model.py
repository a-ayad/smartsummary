import tensorflow as tf
from tensorflow.keras import layers


CNN_FILTERS = 500
CNN_KERNEL_SIZE = 4
EMBEDDING_DIM = 100
DROPOUT = 0.2
LEARNING_RATE = 0.003
BATCH_SIZE = 32
MAX_LENGTH = 2500
EPOCHS = 100


class CNNwLSTM(tf.keras.Model):

    def __init__(self, nb_classes, voc_length, args=None):
        super(CNNwLSTM, self).__init__()
        self.args = vars(args) if args is not None else {}

        filters = self.args.get("cnn_filters", CNN_FILTERS)
        kernel_size = self.args.get("cnn_kernel_size", CNN_KERNEL_SIZE)
        embedding_dim = self.args.get("embedding_dim", EMBEDDING_DIM)
        dropout = self.args.get("dropout", DROPOUT)
        learning_rate = self.args.get("learning_rate", LEARNING_RATE)

        self.embed = layers.Embedding(voc_length, embedding_dim, mask_zero=True)
        self.embed_drop = layers.Dropout(dropout)
        self.conv = layers.Conv1D(filters, kernel_size, padding='valid', activation='relu')
        self.max_pool = layers.MaxPooling1D()
        self.lstm = layers.LSTM(100)
        self.classifier = layers.Dense(nb_classes, activation='sigmoid', name='predictions')

    def call(self, inputs):
        embedded = self.embed(inputs)
        embedded = self.embed_drop(embedded)

        conv_out = self.conv(embedded)
        conv_out = self.max_pool(conv_out)

        lstm_out = self.lstm(conv_out)

        output = self.classifier(lstm_out)

        return output

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--cnn_filters", type=int, default=CNN_FILTERS)
        parser.add_argument("--cnn_kernel_size", type=int, default=CNN_KERNEL_SIZE)
        parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
        parser.add_argument("--epochs", type=int, default=EPOCHS)
        return parser