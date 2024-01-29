import matplotlib.pyplot as plt
from csv import writer
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, hamming_loss, accuracy_score


def load_features(lab=True):
    X_df = pd.read_csv("tsc/data/stacked_input_5.csv")
    if lab:
        FEATURE_COLUMNS = X_df.columns.tolist()[17:-1]
    else:
        FEATURE_COLUMNS = X_df.columns.tolist()[2:-1]
    sequences = []

    for idx, group in X_df.groupby(X_df.index // 5):
        sequence_features = group[FEATURE_COLUMNS]
        sequences.append(sequence_features)

    X = np.array(sequences)
    np.save('/data/features.npy', X)


def load_labels():
    y_df = pd.read_csv("../dataloaders/data/tsc_data/stacked_output_5_numerical.csv")
    FEATURE_COLUMNS = y_df.columns.tolist()[2:-1]
    y = y_df.iloc[:, 2:-1].to_numpy()
    return y, FEATURE_COLUMNS


def plot_epochs_metric(hist, file_name, metric='accuracy'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def load_range():
    lab_range_df = pd.read_csv("../dataloaders/data/tsc_data/lab_ranges_scaled.csv")
    floor = lab_range_df.iloc[0:1, :].to_numpy()
    ceiling = lab_range_df.iloc[1:, :].to_numpy()
    return floor, ceiling


lab_floor, lab_ceiling = load_range()


def transform2binary(y):
    y[(y < lab_floor) | (y > lab_ceiling)] = 1
    y[y != 1] = 0
    y = y.astype(int)
    return y


def acc(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_true = transform2binary(y_true)
    y_pred = transform2binary(y_pred)
    h_loss = hamming_loss(y_true, y_pred)
    acc = 1 - h_loss
    return acc


def show_metrics(y_true, y_pred, feature_names, model_name):
    report = classification_report(y_true, y_pred, target_names=feature_names, output_dict=True)
    exact_match_ratio = accuracy_score(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=feature_names))
    print("Exact Match Ratio: ", exact_match_ratio)
    print("Hamming Loss: ", h_loss)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(model_name + "_classification_report.csv")
    emr = ["Exact Match Ratio:", exact_match_ratio]
    ham_loss = ["Hamming Loss:", h_loss]
    with open(model_name + "_classification_report.csv", 'a') as f_object:
        writer_obj = writer(f_object)
        writer_obj.writerow(emr)
        writer_obj.writerow(ham_loss)
        f_object.close()


