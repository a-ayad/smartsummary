import tensorflow as tf
import pandas as pd
import numpy as np

'''
filepath = 'training/epoch:8-f1:0.554_CNNAttention'
loaded_model = tf.keras.models.load_model(filepath, compile=False)
code_df = pd.read_csv("mimicdata/TOP_50_CODES.csv", header=None)
all_codes = sorted(code_df.iloc[:, 0].tolist())
data_dev_df = pd.read_csv("mimicdata/dev_50.csv")
text_dev = data_dev_df["TEXT"].to_numpy()
codes_dev = data_dev_df["LABELS"].to_numpy()
'''


def vector2code(vector, all_codes):
    t = [all_codes[i] for i, prob in enumerate(vector[0]) if prob > 0.5]
    # codes = [all_codes[i] for i, code in enumerate(vector) if int(code) == 1]
    return t

'''
for index in range(5):
    pred = loaded_model.predict([text_dev[index]])
    pred = vector2code(pred, all_codes)
    print('pred: ', pred)
    print('true: ', codes_dev[index])
'''


def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


if __name__ == '__main__':
    vec = np.array([1, 2, 3, 4, 5])
    print(stable_softmax(vec))
    input()
