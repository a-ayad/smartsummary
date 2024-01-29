import os
import tensorflow.keras as keras
import argparse
from utils.misc import plot_epochs_metric, transform2binary, show_metrics, acc

from tcn import TCN as tcn_nn

DROPOUT = 0.35
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 32
EPOCHS = 50
TCN_FILTERS = 128
TCN_KERNEL_SIZE = 3


class TCN:

    def __init__(self, input_shape, nb_features, verbose=True, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.model = self.build_model(input_shape, nb_features)
        if verbose:
            self.model.summary()
        return

    def build_model(self, input_shape, nb_features):
        learning_rate = self.args.get("learning_rate", LEARNING_RATE)
        tcn_filters = self.args.get("tcn_filters", TCN_FILTERS)
        tcn_kernel_size = self.args.get("tcn_kernel_size", TCN_KERNEL_SIZE)

        input_layer = keras.layers.Input(input_shape)
        tcn_layer = tcn_nn(nb_filters=tcn_filters, kernel_size=tcn_kernel_size, dilations=(1, 2, 4))(input_layer)
        output_layer = keras.layers.Dense(nb_features)(tcn_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mae',
                      optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=[acc],
                      run_eagerly=True)

        return model

    def fit(self, X_train, y_train, X_val, y_val, callbacks):

        mini_batch_size = self.args.get("batch_size", MINI_BATCH_SIZE)
        epochs = self.args.get("epochs", EPOCHS)
        hist = self.model.fit(X_train, y_train,
                              batch_size=mini_batch_size,
                              epochs=epochs,
                              validation_data=(X_val, y_val),
                              callbacks=callbacks)
        self.model.save("tcn.h5")
        plot_epochs_metric(hist, 'tcn_loss', metric='loss')

    def predict(self, X_test, y_true, feature_names):
        # model = keras.models.load_model("tcn.h5")
        model_name = (os.path.basename(__file__)).split('.')[0]
        '''Forecasting below'''
        y_pred = self.model.predict(X_test)
        '''Forecasting above'''

        '''Classification below'''
        y_pred = transform2binary(y_pred)
        y_true = transform2binary(y_true)
        show_metrics(y_true, y_pred, feature_names, model_name)
        '''Classification above'''

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
        parser.add_argument("--batch_size", type=int, default=MINI_BATCH_SIZE)
        parser.add_argument("--epochs", type=int, default=EPOCHS)
        parser.add_argument("--tcn_filters", type=int, default=TCN_FILTERS)
        parser.add_argument("--tcn_kernel_size", type=int, default=TCN_KERNEL_SIZE)
        return parser
