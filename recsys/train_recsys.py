import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import importlib
import argparse
try:
    import wandb
except ModuleNotFoundError:
    pass

from dataloaders.icd_coding_data_loader import load_vocab
from dataloaders.recsys_data_loader import RecSys_DataLoader
from utils import clustering, create_fb_dataset, precision_at_k


PROJECT_NAME = "RecSys"
MODEL_NOTES = None

DATA_FILE = '../dataloaders/data/recsys_data/full_dataV3.csv'
VOCAB_FILE = "../dataloaders/data/icd_coding_data/vocab.csv"
NB_CLUSTERS = 30
SEED = 1


def _import_class(module_and_class_name: str):
    """Import class from a module, e.g. 'models.ModelV2' """
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--model_class", type=str, default="ModelV2")
    parser.add_argument("--nb_clusters", type=int, default=NB_CLUSTERS)
    parser.add_argument("--seed", type=int, default=SEED)

    # Get the model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    model_class = _import_class(f"models.{temp_args.model_class}")

    # Get model specific arguments
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    # Arguments for Project name and Notes on the weights and biases tool
    parser.add_argument("-n", "--notes", type=str, default=MODEL_NOTES, help="Notes about the run")
    parser.add_argument("-p", "--project_name", type=str, default=PROJECT_NAME, help="Main project name")

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    vocab = load_vocab(VOCAB_FILE)

    if args.model_class == "ModelV1":
        version = 'v1'
    else:
        version = 'v2'

    recsys_dataloader = RecSys_DataLoader(DATA_FILE, version)
    data_dict, lab_ratings = recsys_dataloader.get_data()
    user_age = data_dict['age']
    user_weight_kg = data_dict['weight']
    disease = data_dict['disease']
    icd_codes = data_dict['codes']

    user_age_train, user_age_val, user_weight_kg_train, user_weight_kg_val = train_test_split(user_age, user_weight_kg, random_state=42, test_size=0.1)
    disease_train, disease_val, icd_codes_train, icd_codes_val = train_test_split(disease, icd_codes, random_state=42, test_size=0.1)
    lab_ratings_train, lab_ratings_val, _, _ = train_test_split(lab_ratings, user_age, random_state=42, test_size=0.1)

    model_class = _import_class(f"models.{args.model_class}")
    model = model_class(vocab=vocab, user_age=user_age, user_weight_kg=user_weight_kg)
    model.compile('rmsprop', 'mse', run_eagerly=True)
    model.fit((user_age, user_weight_kg, icd_codes, disease), lab_ratings, batch_size=64, epochs=1)
    print(model.summary())
    model.compile('rmsprop', 'mse', run_eagerly=False)
    # model.fit(data_dict, lab_ratings, batch_size=64, epochs=50)
    model.fit((user_age, user_weight_kg, icd_codes, disease), lab_ratings, batch_size=64, epochs=50)

    if version == 'v1':
        # V1
        ic_embed_mean = model.icd_embedding.predict(icd_codes).mean(axis=-1)

    dis_embed_mean = model.disease_embeddings.predict(disease).mean(axis=-1)

    fb_df = recsys_dataloader.get_data_df().copy()

    # fb_df['ic_embed_mean'] = ic_embed_mean
    fb_df['dis_embed_mean'] = dis_embed_mean

    if version == 'v2':
        # V2
        features = np.stack((user_age, user_weight_kg, dis_embed_mean), axis=1)
        for i in range(icd_codes.shape[1]):
            features = np.hstack((features, icd_codes[:, [i]]))
    else:
        # V1
        features = np.stack((user_age, user_weight_kg, dis_embed_mean, ic_embed_mean), axis=1)

    nb_clusters = args.nb_clusters
    cluster_labels = clustering(features=features, nb_clusters=nb_clusters)
    fb_df['cluster_labels'] = cluster_labels

    fb_data_train, fb_labels_train, \
    fb_data_test, fb_labels_test, \
    fb_test_data_ls, fb_test_labels_ls = create_fb_dataset(fb_df=fb_df, nb_clusters=nb_clusters, version=version)

    # top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    # top5_acc.reset_state()
    # top5_acc.update_state(fb_labels_test, model.predict((fb_data_test['age'],
    #                                                      fb_data_test['weight'],
    #                                                      fb_data_test['codes'],
    #                                                      fb_data_test['disease'])))
    # print(f"Before Feedback dataset Acc ALL: {top5_acc.result().numpy()}")
    pred_score_bf = model.predict((fb_data_test['age'],
                                   fb_data_test['weight'],
                                   fb_data_test['codes'],
                                   fb_data_test['disease']))
    topk = 5
    print(f"Before Feedback dataset Precision@{topk}: {precision_at_k(fb_labels_test, pred_score_bf, topk)}")
    # (user_age, user_weight_kg, icd_codes, disease)
    sample_w = np.full((fb_labels_train.shape[0], 1), 1.5)
    if args.wandb:
        # initialize wandb logging to your project
        wandb.init(project=args.project_name, notes=args.notes)
        # log all experimental args to wandb
        wandb.config.update(args)
        # ckpt_wandb = tf.keras.callbacks.ModelCheckpoint(f"{wandb.run.dir}/best_model", monitor="val_f1_score_micro", mode='max', verbose=0, save_best_only=True, save_weights_only=True)
        callbacks = [wandb.keras.WandbCallback()]
    else:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1,
                                                         patience=5, verbose=1)
        callbacks = [reduce_lr]

    model.fit((fb_data_train['age'],
               fb_data_train['weight'],
               fb_data_train['codes'],
               fb_data_train['disease']), fb_labels_train, batch_size=32, epochs=20, sample_weight=sample_w,
              validation_data=((fb_data_test['age'],
                                fb_data_test['weight'],
                                fb_data_test['codes'],
                                fb_data_test['disease']), fb_labels_test),
              callbacks=callbacks)

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    pred_score_after = model.predict((fb_data_test['age'],
                                   fb_data_test['weight'],
                                   fb_data_test['codes'],
                                   fb_data_test['disease']))

    print(f"After training feedback dataset Precision@{topk}: {precision_at_k(fb_labels_test, pred_score_after, topk)}")

    for i in range(nb_clusters):
        pred_score = model.predict((fb_test_data_ls[i]['age'],
                                    fb_test_data_ls[i]['weight'],
                                    fb_test_data_ls[i]['codes'],
                                    fb_test_data_ls[i]['disease']))

        print(f"Group {i} Precision@{topk}: {precision_at_k(fb_test_labels_ls[i], pred_score, topk)}")
    # top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    # top5_acc.reset_state()
    # top5_acc.update_state(fb_labels_test, model.predict((fb_data_test['age'],
    #                                                      fb_data_test['weight'],
    #                                                      fb_data_test['codes'],
    #                                                      fb_data_test['disease'])))
    # print(f"Acc ALL: {top5_acc.result().numpy()}")

    # for i in range(nb_clusters):
    #     top5_acc.reset_state()
    #     top5_acc.update_state(fb_test_labels_ls[i], model.predict((fb_test_data_ls[i]['age'],
    #                                                                fb_test_data_ls[i]['weight'],
    #                                                                fb_test_data_ls[i]['codes'],
    #                                                                fb_test_data_ls[i]['disease'])))
    #     print(f"Group {i}: {top5_acc.result().numpy()}")


if __name__ == '__main__':
    main()



