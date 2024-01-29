import os
import tensorflow.keras as keras
import argparse
from utils.misc import plot_epochs_metric, transform2binary, show_metrics, acc


LSTM_UNIT = 128
DROPOUT = 0.35
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 32
EPOCHS = 50

MODEL_NAME = (os.path.basename(__file__)).split('.')[0]


class LSTM:

    def __init__(self, input_shape, nb_features, verbose=True, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.model = self.build_model(input_shape, nb_features)
        if verbose:
            self.model.summary()
        return

    def build_model(self, input_shape, nb_features):

        lstm_unit = self.args.get("lstm_unit", LSTM_UNIT)
        dropout = self.args.get("dropout", DROPOUT)
        learning_rate = self.args.get("learning_rate", LEARNING_RATE)

        input_layer = keras.layers.Input(input_shape)
        lstm1 = keras.layers.LSTM(units=lstm_unit)(input_layer)
        lstm1 = keras.layers.Dropout(rate=dropout)(lstm1)

        output_layer = keras.layers.Dense(units=nb_features)(lstm1)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mae',
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[acc],
                      run_eagerly=True)

        return model

    def fit(self, X_train, y_train, X_val, y_val, callbacks=None):

        mini_batch_size = self.args.get("batch_size", MINI_BATCH_SIZE)
        epochs = self.args.get("epochs", EPOCHS)
        hist = self.model.fit(X_train, y_train,
                              batch_size=mini_batch_size,
                              epochs=epochs,
                              validation_data=(X_val, y_val),
                              callbacks=callbacks)
        self.model.save("lstm.h5")
        plot_epochs_metric(hist, 'lstm_loss', metric='loss')

    def predict(self, X_test, y_true, feature_names):
        # model = keras.models.load_model("lstm.h5")
        '''Forecasting below'''
        y_pred = self.model.predict(X_test)
        '''Forecasting above'''

        '''Classification below'''
        y_pred = transform2binary(y_pred)
        y_true = transform2binary(y_true)
        show_metrics(y_true, y_pred, feature_names, MODEL_NAME)
        '''Classification above'''

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lstm_unit", type=int, default=LSTM_UNIT)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=MINI_BATCH_SIZE)
        parser.add_argument("--epochs", type=int, default=EPOCHS)
        return parser
