import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def colorize(words, color_array):
    cmap = matplotlib.cm.Reds
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        # print(color)
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    return colored_string


def vector2code(vector, all_codes, prob=False, return_index=False):
    if prob:
        codes = [all_codes[i] for i, prob in enumerate(vector[0]) if prob > 0.5]
        if return_index:
            codes = [[all_codes[i], i] for i, prob in enumerate(vector[0]) if prob > 0.5]
    else:
        codes = [all_codes[i] for i, code in enumerate(vector) if int(code) == 1]
    return codes


if __name__ == '__main__':

    filepath = 'training/epoch:8-f1:0.554_CNNAttention'
    loaded_model = tf.keras.models.load_model(filepath, compile=False)
    model = tf.keras.models.Model(inputs=loaded_model.input,
                                  outputs=[loaded_model.output, loaded_model.get_layer('attention').output[0]])
    # output = model.predict(['test_text'])
    # prediction, attention_weights = output
    data_dev_df = pd.read_csv("mimicdata/dev_50.csv")
    text_dev = data_dev_df["TEXT"].to_numpy()
    codes_dev = data_dev_df["LABELS"].to_numpy()
    code_df = pd.read_csv("mimicdata/TOP_50_CODES.csv", header=None)
    all_codes = sorted(code_df.iloc[:, 0].tolist())
    '''
    test_text = text_dev[0]
    sent = test_text.split()
    sent_length = len(sent)
    output = model.predict([test_text])
    prediction, attention_weights = output
    attention_weights = attention_weights[0]
    weights4code1 = attention_weights[0][:sent_length]
    att = colorize(sent, weights4code1)
    for index in range(len(text_dev)):
        test_text = text_dev[0]
        sent = test_text.split()
        sent_length = len(sent)
        output = model.predict([test_text])
        prediction, attention_weights = output
        attention_weights = attention_weights[0]
        weights4code1 = attention_weights[0][:sent_length]
        att = colorize(sent, weights4code1)
    '''
    '''
    for index in range(10):
        print('split: ', len(text_dev[index].split()))
    '''
    '''
    voc_df = pd.read_csv("mimicdata/vocab.csv", header=None)
    vocab = sorted(voc_df.iloc[:, 0].tolist())
    vectorize_layer_voc = TextVectorization(standardize="lower_and_strip_punctuation",
                                                 max_tokens=len(vocab) + 2,
                                                 output_mode='int',
                                                 output_sequence_length=2500)
    text_dataset = tf.data.Dataset.from_tensor_slices(vocab)
    vectorize_layer_voc.adapt(text_dataset.batch(64))
    model_c = tf.keras.models.Sequential()
    model_c.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model_c.add(vectorize_layer_voc)
    # code_vector = model_c.predict(code)
    '''

    words = 'The quick brown fox jumps over the lazy dog'.split()
    color_array = np.random.rand(len(words))
    # print(color_array)
    s = colorize(words, color_array)

    # or simply save in an html file and open in browser
    with open('attention.html', 'w') as f:
        for index in range(10):
            test_text = text_dev[index]
            sent = test_text.split()
            sent_length = len(sent)
            output = model.predict([test_text])
            prediction, attention_weights = output
            attention_weights = attention_weights[0]

            preds = vector2code(prediction, all_codes, prob=True, return_index=True)
            true = codes_dev[index]
            f.write("<pre>" + "Sample " + str(index) + "</pre>\n")
            for pred in preds:
                weights4code = attention_weights[pred[1]][:sent_length]
                att = colorize(sent, weights4code)
                f.write("<pre>" + str(pred[0]) + "</pre>\n")
                f.write("<pre>" + att + "</pre>\n")

            f.write("<pre>" + "Prediction: " + str(preds) + "</pre>\n")
            f.write("<pre>" + "Ground Truth: " + str(true) + "</pre>\n")

            # weights4code1 = attention_weights[i][:sent_length]
            # att = colorize(sent, weights4code1)

