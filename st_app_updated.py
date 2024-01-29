import streamlit as st
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from disease_name_recog.data.dataset_preprocessing import fetch_sentences
from disease_name_recog.model.biobert_ner import sparse_categorical_accuracy_masked, sparse_crossentropy_masked
from dataloaders.recsys_data_loader import RecSys_DataLoader
import icd_coding.models as md
from dataloaders.icd_coding_data_loader import load_vocab

from datetime import datetime, timedelta
import run_recsys

FILEPATH = 'dataloaders/data/recsys_data/full_dataV3.csv'
VOCAB_FILE = "dataloaders/data/icd_coding_data/vocab.csv"
RECSYS_MODEL_PATH = 'recsys/recsys_model'
RECSYS_OG_MODEL_PATH = 'recsys/recsys_og'

ICD_MODEL_PATH = "icd_coding/training/CNN_Nov30/best_model"
BIOBERT_VOCAB_PATH = "disease_name_recog/biobert_v1.1_pubmed/vocab.txt"
BIOBERT_NER_MODEL_PATH = 'disease_name_recog/model/biobert_ner'

LAB_DF_FILEPATH = "dataloaders/data/tsc_data/stacked_input_5.csv"

ICD_DESC_FILEPATH = "ICD9_descriptions"

test_hadmid = [
    "105131 (1)",
    "147038 (1)",
    "150158 (1)",
    "125199 (2)",
    "127513 (2)",
    "166879 (2)",
    "121000 (3)",
    "147061 (3)",
    "159596 (3)",
]

hadm_id_list = [
    105131,
    147038,
    150158,
    125199,
    127513,
    166879,
    121000,
    147061,
    159596
]

LAB_NAMES = [
    "Albumin [g/dL]",
    "Arterial Base Excess [mEq/L]",
    "Arterial PH",
    "Blood urea nitrogen [mg/dL]",
    "Calcium [mEq/L]",
    "Chloride [mEq/L]",
    "CO2 [mEq/L]",
    "Creatinine [mg/dL]",
    "Glucose [mg/dL]",
    "Hemoglobin [g/dl]",
    "Bicarbonate [mEq/L]",
    "International Normalised Ratio",
    "Lactate [mmol/L]",
    "Magnesium [mg/dL]",
    "PaCO2 [mmHg]",
    "PaO2 [mmHg]",
    "PaO2/FiO2 Ratio",
    "Platelets Count [x10^9/L]",
    "Potassium [mEq/L]",
    "Prothrombin time [S]",
    "Partial Thromboplastin Time [S]",
    "Sodium [mEq/L]",
    "SpO2 [%]",
    "Total Bilirubin [mg/dL]",
    "WBC Count [x10^9/L]"
]


@st.cache
def load_range():
    lab_range_df = pd.read_csv("dataloaders/data/tsc_data/lab_ranges_scaled.csv")
    floor = lab_range_df.iloc[0:1, :].to_numpy()
    ceiling = lab_range_df.iloc[1:, :].to_numpy()
    return floor, ceiling


def transform2binary(y, lab_floor, lab_ceiling):
    y[(y < lab_floor) | (y > lab_ceiling)] = 1
    y[y != 1] = 0
    y = y.astype(int)
    return y


@st.cache
def load_recsys_dataloder(filepath):
    recsys_dataloader = RecSys_DataLoader(filepath)
    return recsys_dataloader


# @st.cache
def load_recsys_model(model_path):
    recsys_model = tf.keras.models.load_model(model_path)
    return recsys_model

# @st.cache
def load_recsys_og_model(model_path):
    recsys_og_model = tf.keras.models.load_model(model_path)
    return recsys_og_model


@st.cache
def load_biobert_tokenizer(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer


@st.cache
def load_biobert_model(model_path):
    biobert_model = tf.keras.models.load_model(model_path,
                                               custom_objects={
                                                   'sparse_categorical_accuracy_masked': sparse_categorical_accuracy_masked,
                                                   'sparse_crossentropy_masked': sparse_crossentropy_masked})
    return biobert_model


@st.cache
def load_icd_coding_model(vocab, model_weights_path):
    icd_coding_model = md.CNN(vocab=vocab)
    icd_coding_model = icd_coding_model.build_model(50)
    icd_coding_model.load_weights(model_weights_path)
    return icd_coding_model


@st.cache
def load_dataframe(df, list_of_hadmid):
    data_df = df[df['HADM_ID'].isin(list_of_hadmid)]
    return data_df


@st.cache
def load_lab_df(filepath):
    lab_values_df = pd.read_csv(filepath)
    return lab_values_df


@st.cache
def load_icd_desc(filepath, included_codes):
    desc = pd.read_csv(filepath, sep="\t", header=None)
    desc.columns = ["ICD_CODES", "DESC"]
    desc = desc[desc['ICD_CODES'].isin(included_codes)]
    return desc


def color_abnormal(v, color='red'):
    return f"color: {color};" if v == 1 else None


if __name__ == '__main__':
    lab_floor, lab_ceiling = load_range()

    recsys_dataloader = RecSys_DataLoader(FILEPATH)
    data_df = recsys_dataloader.get_data_df()
    vocab = load_vocab(VOCAB_FILE)
    top_50_codes = load_vocab("dataloaders/data/icd_coding_data/TOP_50_CODES.csv")

    icd_coding_model = load_icd_coding_model(vocab=vocab, model_weights_path=ICD_MODEL_PATH)

    tokenizer = load_biobert_tokenizer(BIOBERT_VOCAB_PATH)
    biobert_model = load_biobert_model(BIOBERT_NER_MODEL_PATH)

    recsys = load_recsys_model(RECSYS_MODEL_PATH)

    recsys_og = load_recsys_og_model(RECSYS_OG_MODEL_PATH)

    data_df = load_dataframe(data_df, hadm_id_list)
    st.title("Smart Summary")
    hadm_id_selected = st.selectbox('Patient ID:', test_hadmid)
    hadm_id = int(hadm_id_selected[:6])

    user_notes = data_df[data_df['HADM_ID'] == hadm_id]['TEXT'].values
    user_age = data_df[data_df['HADM_ID'] == hadm_id]['age'].values
    user_weight = data_df[data_df['HADM_ID'] == hadm_id]['Weight_kg'].values
    icustay_id = data_df[data_df['HADM_ID'] == hadm_id]['icustayid'].item()

    user_icd_codes_binary = np.asarray(icd_coding_model.predict(user_notes)).round()
    user_icd_codes = run_recsys.vector2code(user_icd_codes_binary[0], top_50_codes)

    user_disease_name = str(list(set(run_recsys.get_disease(user_notes[0])[0])))

    user_profile_dict = {"hadm_id": hadm_id,
                         "codes": user_icd_codes_binary,
                         "age": user_age,
                         "weight": user_weight,
                         "disease": user_disease_name}
    user_profile = (user_profile_dict['age'],
                    user_profile_dict['weight'],
                    user_profile_dict['codes'],
                    np.expand_dims(user_profile_dict['disease'], axis=0))
    click_state = dict()

    if hadm_id not in st.session_state:
        st.session_state[hadm_id] = 0
    else:
        st.session_state[hadm_id] = 1

    if st.session_state[hadm_id]:
        lab_ratings_pred = recsys.predict(user_profile)[0]
        my_expander = st.expander(label="Explainability")
        sim_patients = []
        for p in test_hadmid:
            if hadm_id_selected != p and hadm_id_selected[6:] in p:
                sim_patients.append(p[:6])
        sim_explain_str_output = f"Patient {hadm_id} is similar to "
        for sim_p in sim_patients:
            sim_explain_str_output += f'{sim_p}, '
        with my_expander:
            sim_explain_str_output
    else:
        lab_ratings_pred = recsys_og.predict(user_profile)[0]
        my_expander = st.expander(label="Explainability")
        with my_expander:
            "Abnormal lab values should be concerned"

    top5_lab = np.argsort(lab_ratings_pred)[:5]

    lab_values_df = load_lab_df(LAB_DF_FILEPATH)

    user_lab_values = lab_values_df[lab_values_df['icustayid'] == icustay_id].iloc[-5:].iloc[:,
                      17:-1].to_numpy()
    user_binary = np.copy(user_lab_values)
    user_binary = transform2binary(user_binary, lab_floor, lab_ceiling)

    lab_pair = list(zip(LAB_NAMES, user_lab_values.T, user_binary.T))
    lab_dict_key = [x for x in range(len(lab_pair))]
    lab_dict = dict(zip(lab_dict_key, lab_pair))
    res = [lab_dict[x] for x in top5_lab]
    res_all = [lab_dict[x] for x in np.argsort(lab_ratings_pred)]

    user_profile_dict['codes'] = user_icd_codes

    st.write(f"Diagnosis: {user_disease_name}")
    # st.write(f"Predicted ICD codes: {user_icd_codes}")

    desc_df = load_icd_desc(filepath=ICD_DESC_FILEPATH, included_codes=top_50_codes)
    desc_dict = dict(zip(desc_df.ICD_CODES, desc_df.DESC))

    st.write("Predicted ICD Codes")
    for icd_code in user_icd_codes:
        my_expander = st.expander(label=icd_code)
        with my_expander:
            desc_dict[icd_code]

    st.write("Predicted Length of Stay: ")
    current_time = datetime.now()
    delta_time = timedelta(hours=4)
    my_dataframe = pd.DataFrame({
        'Items': [i[0] for i in res],
        f'{(current_time - 3*delta_time).strftime("%H:%M")}': [i[1][0] for i in res],
        f'{(current_time - 2*delta_time).strftime("%H:%M")}': [i[1][1] for i in res],
        f'{(current_time - delta_time).strftime("%H:%M")}': [i[1][2] for i in res],
        f'{current_time.strftime("%H:%M")}': [i[1][3] for i in res],
        f'{(current_time + delta_time).strftime("%H:%M")}': [i[1][4] for i in res],
    })

    binary_dataframe = pd.DataFrame({
        'Items': [i[0] for i in res],
        f'{(current_time - 3*delta_time).strftime("%H:%M")}': [i[2][0] for i in res],
        f'{(current_time - 2*delta_time).strftime("%H:%M")}': [i[2][1] for i in res],
        f'{(current_time - 1*delta_time).strftime("%H:%M")}': [i[2][2] for i in res],
        f'{current_time.strftime("%H:%M")}': [i[2][3] for i in res],
        f'{(current_time + delta_time).strftime("%H:%M")}': [i[2][4] for i in res],
    })

    st.write("Lab values: ")
    col1, col2 = st.columns([3, 1])

    col1.table(my_dataframe.style.format({f'{(current_time - 3*delta_time).strftime("%H:%M")}': "{:.2f}",
                                          f'{(current_time - 2*delta_time).strftime("%H:%M")}': "{:.2f}",
                                          f'{(current_time - 1*delta_time).strftime("%H:%M")}': "{:.2f}",
                                          f'{current_time.strftime("%H:%M")}': "{:.2f}",
                                          f'{(current_time + delta_time).strftime("%H:%M")}': "{:.2f}"})\
            .apply(lambda x: (binary_dataframe.applymap(color_abnormal)), axis=None))
    for i in res:
        col2.number_input(i[0], min_value=1, max_value=10, value=10, step=1)

    my_dataframe_all = pd.DataFrame({
        'Items': [i[0] for i in res_all],
        f'{(current_time - 3*delta_time).strftime("%H:%M")}': [i[1][0] for i in res_all],
        f'{(current_time - 2*delta_time).strftime("%H:%M")}': [i[1][1] for i in res_all],
        f'{(current_time - 1*delta_time).strftime("%H:%M")}': [i[1][2] for i in res_all],
        f'{current_time.strftime("%H:%M")}': [i[1][3] for i in res_all],
        f'{(current_time + delta_time).strftime("%H:%M")}': [i[1][4] for i in res_all],
    })

    binary_dataframe_all = pd.DataFrame({
        'Items': [i[0] for i in res_all],
        f'{(current_time - 3*delta_time).strftime("%H:%M")}': [i[2][0] for i in res_all],
        f'{(current_time - 2*delta_time).strftime("%H:%M")}': [i[2][1] for i in res_all],
        f'{(current_time - 1*delta_time).strftime("%H:%M")}': [i[2][2] for i in res_all],
        f'{current_time.strftime("%H:%M")}': [i[2][3] for i in res_all],
        f'{(current_time + delta_time).strftime("%H:%M")}': [i[2][4] for i in res_all],
    })

    my_expander = st.expander(label="Expand Me for all lab values")
    with my_expander:
        col1, col2 = st.columns([3, 1])
        col1.table(my_dataframe_all.style.format({f'{(current_time - 3*delta_time).strftime("%H:%M")}': "{:.2f}",
                                                  f'{(current_time - 2*delta_time).strftime("%H:%M")}': "{:.2f}",
                                                  f'{(current_time - 1*delta_time).strftime("%H:%M")}': "{:.2f}",
                                                  f'{current_time.strftime("%H:%M")}': "{:.2f}",
                                                  f'{(current_time + delta_time).strftime("%H:%M")}': "{:.2f}"})\
                   .apply(lambda x: (binary_dataframe_all.applymap(color_abnormal)), axis=None))
        for i in res_all:
            col2.number_input(i[0], min_value=1, max_value=10, value=5, step=1, key=i)
    print(res[0][1][4])
    print(my_dataframe)

