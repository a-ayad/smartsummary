import os
import tensorflow.keras as keras
import argparse
from utils.misc import plot_epochs_metric, transform2binary, show_metrics, acc

DROPOUT = 0.35
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 32
EPOCHS = 50


class FCN:

    def __init__(self, input_shape, nb_features, verbose=True, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.model = self.build_model(input_shape, nb_features)
        if verbose:
            self.model.summary()
        return

    def build_model(self, input_shape, nb_features):
        learning_rate = self.args.get("learning_rate", LEARNING_RATE)

        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_features)(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mse',
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
        self.model.save("fcn.h5")
        plot_epochs_metric(hist, 'fcn_loss', metric='loss')

    def predict(self, X_test, y_true, feature_names):
        # model = keras.models.load_model("fcn.h5")
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
        return parser


