import pandas as pd
import numpy as np
import tensorflow as tf


class RecSys_DataLoader:
    def __init__(self, filepath, version='v2'):
        data_df = pd.read_csv(filepath)
        data_df = data_df.drop(data_df[data_df['disease'].map(len) < 3].index)
        self.data_df = data_df.drop(data_df[data_df["icd_codes_labels"] != data_df["icd_codes_labels"]].index)

        self.lab_ratings = self.data_df.iloc[:, 2:-7].to_numpy()
        # data['TEXT'] = data['TEXT'].apply(clean)

        text = self.data_df['TEXT'].to_numpy()
        hadm_id = self.data_df['HADM_ID'].to_numpy(dtype=str)
        unique_hadm_id = np.unique(hadm_id)

        user_age = self.data_df['age'].to_numpy()
        user_weight_kg = self.data_df['Weight_kg'].to_numpy()
        disease = self.data_df['disease'].to_numpy()
        if version == 'v2':
            # V2
            icd_codes = np.asarray([toarr(clean_icd_labels(x)) for x in self.data_df['icd_codes']])
        else:
            # v1
            icd_codes = np.asarray([toarr(clean_icd_labels(x)) for x in self.data_df['icd_codes_labels']])
            icd_codes = tf.keras.preprocessing.sequence.pad_sequences(icd_codes, padding='post')
        self.data_dict = {"hadm_id": hadm_id,
                          "codes": icd_codes,
                          "age": user_age,
                          "weight": user_weight_kg,
                          "disease": disease}

    def get_data(self):
        return self.data_dict, self.lab_ratings

    def get_data_df(self):
        return self.data_df


def toarr(label):
    return np.fromstring(label, sep=' ').astype('float32')


def clean_icd_labels(icd_label):
    res = icd_label.replace("[", "")
    res = res.replace("]", "")
    res = res.strip()
    return res