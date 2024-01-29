import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import operator
from load_data import load_code_descriptions


def save_samples(data, output, target_data, s, filter_size, tp_file, fp_file, dicts=None):
    """
        save important spans of text from attention
        INPUTS:
            data: input data (text) to the model
            output: model prediction
            target_data: ground truth labels
            s: attention vector from attn model
            filter_size: size of the convolution filter, and length of phrase to extract from source text
            tp_file: opened file to write true positive results
            fp_file: opened file to write false positive results
            dicts: hold info for reporting results in human-readable form
    """
    tgt_codes = np.where(target_data[0] == 1)[0]
    true_str = "Y_true: " + str(tgt_codes)
    output_rd = np.round(output)
    pred_codes = np.where(output_rd[0] == 1)[0]
    pred_str = "Y_pred: " + str(pred_codes)
    if dicts is not None:
        if s is not None and len(pred_codes) > 0:
            important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, tp_file, fps=False)
            important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, fp_file, fps=True)


def important_spans(data, output, tgt_codes, pred_codes, s, ind2w, ind2c, desc_dict, filter_size, true_str, pred_str, spans_file, fps=False):
    """
        looks only at the first instance in the batch
    """
    # ind2w, ind2c, desc_dict = dicts['ind2w'], dicts['ind2c'], dicts['desc']
    for p_code in pred_codes:
        #aww yiss, xor... if false-pos mode, save if it's a wrong prediction, otherwise true-pos mode, so save if it's a true prediction
        if output[0][p_code] > .5 and (fps ^ (p_code in tgt_codes)):
            confidence = output[0][p_code]

            #some info on the prediction
            code = ind2c[p_code]
            conf_str = "confidence of prediction: %f" % confidence
            typ = "false positive" if fps else "true positive"
            prelude = "top three important windows for %s code %s (%s: %s)" % (typ, str(p_code), code, desc_dict[code])

            if spans_file is not None:
                spans_file.write(conf_str + "\n")
                spans_file.write(true_str + "\n")
                spans_file.write(pred_str + "\n")
                spans_file.write(prelude + "\n")

            #find most important windows
            attn = s[p_code]
            #merge overlapping intervals
            imps = attn.argsort()[-10:][::-1]
            windows = make_windows(imps, filter_size, attn)
            kgram_strs = []
            i = 0
            while len(kgram_strs) < 3 and i < len(windows):
                (start, end), score = windows[i]
                words = [ind2w[w] if w in ind2w.keys() else 'UNK' for w in data[start:end]]
                kgram_str = " ".join(words) + ", score: " + str(score)
                #make sure the span is unique
                if kgram_str not in kgram_strs:
                    kgram_strs.append(kgram_str)
                i += 1
            for kgram_str in kgram_strs:
                if spans_file is not None:
                    spans_file.write(kgram_str + "\n")
            spans_file.write('\n')

def make_windows(starts, filter_size, attn):
    starts = sorted(starts)
    windows = []
    overlaps_w_next = [starts[i+1] < starts[i] + filter_size for i in range(len(starts)-1)]
    overlaps_w_next.append(False)
    i = 0
    get_new_start = True
    while i < len(starts):
        imp = starts[i]
        if get_new_start:
            start = imp
        overlaps = overlaps_w_next[i]
        if not overlaps:
            windows.append((start, imp+filter_size))
        get_new_start = not overlaps
        i += 1
    #return windows sorted by decreasing importance
    window_scores = {(start, end): attn[start] for (start, end) in windows}
    window_scores = sorted(window_scores.items(), key=operator.itemgetter(1), reverse=True)
    return window_scores


def transform_code2vector(code, all_codes):

    def split_by_semicolon(inputs):
        return tf.strings.split(inputs, sep=';')

    vectorize_layer_code = TextVectorization(standardize=None,
                                             max_tokens=len(all_codes)+1,
                                             output_mode='binary',
                                             vocabulary=all_codes,
                                             split=split_by_semicolon)
    model_c = tf.keras.models.Sequential()
    model_c.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model_c.add(vectorize_layer_code)
    code_vector = model_c.predict(code)
    # Delete PAD
    labels = np.delete(code_vector, 0, -1)
    return labels, vectorize_layer_code.get_vocabulary()


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
    textVector_layer = loaded_model.get_layer("text_vectorization_2")
    vocab = textVector_layer.get_vocabulary()

    desc_dict = load_code_descriptions()
    code_df = pd.read_csv("mimicdata/TOP_50_CODES.csv", header=None)
    all_codes = sorted(code_df.iloc[:, 0].tolist())

    ind2word = {index: word for index, word in enumerate(vocab)}
    ind2code = {index: code for index, code in enumerate(all_codes)}

    data_dev_df = pd.read_csv("mimicdata/dev_50.csv")
    text_dev = data_dev_df["TEXT"].to_numpy()
    codes_dev = data_dev_df["LABELS"].to_numpy()
    labels_dev, _ = transform_code2vector(codes_dev, all_codes)
    tp_file = open('explain_tp.txt', 'w')
    fp_file = open('explain_fp.txt', 'w')
    for indx in range(len(text_dev)):
        print("Sample: %s" % indx)
        test_text = text_dev[indx]
        sent = test_text.split()
        sent_length = len(sent)
        text_vec = (textVector_layer([test_text])[0].numpy())[:sent_length]
        tgt_codes = np.where(labels_dev[indx] == 1)[0]
        output = model.predict([test_text])
        prediction, attention_weights = output
        attention_weights = attention_weights[0][:, :sent_length]
        output_rd = np.round(prediction)
        pred_codes = np.where(output_rd[0] == 1)[0]
        true_str = "Y_true: " + str(tgt_codes)
        pred_str = "Y_pred: " + str(pred_codes)
        # preds = vector2code(prediction, all_codes, prob=True, return_index=True)
        if len(pred_codes) > 0:
            important_spans(text_vec, prediction, tgt_codes, pred_codes, attention_weights, ind2word, ind2code, desc_dict,
                            4, true_str, pred_str, tp_file, fps=False)
            important_spans(text_vec, prediction, tgt_codes, pred_codes, attention_weights, ind2word, ind2code, desc_dict,
                            4, true_str, pred_str, fp_file, fps=True)
    tp_file.close()
    fp_file.close()
    print('done')
    input()



