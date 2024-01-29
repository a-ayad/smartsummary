
class BaseModel(object):
    def __init__(self, vocab, args=None, verbose=True):
        self.model = None

    def evaluate(self, X, y, batch_size=None):
        return self.model.evaluate(X, y, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self, output_shape):
        raise NotImplementedError