import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class RecSysDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.ratings = y

    def __getitem__(self, index):
        age = self.data['age'][index]
        weight = self.data['weight'][index]
        icd_codes = self.data['codes'][index]
        disease = self.data['encoded_disease'][index]
        target = self.ratings[index]
        return {
            'age': torch.tensor(age, dtype=float),
            'weight': torch.tensor(weight, dtype=float),
            'disease': torch.tensor(disease, dtype=torch.long),
            'icd_codes': torch.tensor(icd_codes, dtype=float),
            'target': torch.tensor(target, dtype=float)
        }

    def __len__(self):
        return len(self.data['age'])

    def create_dataloader(self, batch_size, shuffle):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle
        )


def preprocess(patient_dict, w2idx, num_traindata, client_order):
    encoded_disease = encoding_disease(patient_dict, w2idx)
    encoded_disease_padded = pad_sequences(encoded_disease)
    patient_dict['encoded_disease'] = encoded_disease_padded

    patient_dict['age'] = patient_dict['age'].reshape((-1, 1))
    patient_dict['weight'] = patient_dict['weight'].reshape((-1, 1))

    scaler_age = MinMaxScaler()
    scaler_weight = MinMaxScaler()

    if num_traindata and client_order:
        patient_dict['age'] = scaler_age.fit_transform(patient_dict['age'])[num_traindata * client_order: num_traindata * (client_order + 1)]
        patient_dict['weight'] = scaler_weight.fit_transform(patient_dict['weight'])[num_traindata * client_order: num_traindata * (client_order + 1)]
        patient_dict['codes'] = patient_dict['codes'][num_traindata * client_order: num_traindata * (client_order + 1)]

    patient_dict['age'] = scaler_age.fit_transform(patient_dict['age'])
    patient_dict['weight'] = scaler_weight.fit_transform(patient_dict['weight'])

    return patient_dict


def encoding_disease(data, w2idx):
    idx_total = []
    for i in range(len(data['age'])):
        text = data['disease'][i]
        cleaned_text = clean_text(text)
        idx = []
        for st in cleaned_text:
            if st not in w2idx:
                idx.append(len(w2idx) + 1)
            else:
                idx.append(w2idx[st])
        idx_total.append(idx)
    return np.array(idx_total)


def clean_text(text):
    s = text.replace('[', "")
    s = s.replace(']', "")
    s = s.replace("'", "")
    s = s.replace(",", "")
    s = s.split()
    return s


def load_vocab_dict(vocab_file):
    vocab_df = pd.read_csv(vocab_file, header=None)
    vocab = sorted(set(vocab_df[0].tolist()))
    ind2w = {i+1:w for i,w in enumerate(vocab)}
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind


def pad_sequences(sequences, max_seq_len: int = 0):
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    # Pad
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][: len(sequence)] = sequence
    return padded_sequences


