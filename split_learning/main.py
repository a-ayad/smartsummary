import pickle
import socket
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from split_learning import data, models, utils, train

FILEPATH_DATA = 'data.p'
FILEPATH_RATINGS = 'lab_ratings.p'
FILEPATH_FB_DATA_TRAIN = "fb_data_train.p"
FILEPATH_FB_LABELS_TRAIN = "fb_labels_train.p"
FILEPATH_FB_DATA_TEST = "fb_data_test.p"
FILEPATH_FB_LABELS_TEST = "fb_labels_test.p"

FILEPATH_VOCAB = 'vocab.csv'
EMBED_FILE = 'processed_full.embed'

EPOCHS = 50


def load_data(pickle_file):
    with open(pickle_file, "rb") as openfile:
        data = pickle.load(openfile)
    return data


def train_server(users):
    device = "cpu"

    train_data = load_data(FILEPATH_DATA)
    lab_ratings = load_data(FILEPATH_RATINGS)
    fb_data_train = load_data(FILEPATH_FB_DATA_TRAIN)
    fb_labels_train = load_data(FILEPATH_FB_LABELS_TRAIN)
    fb_data_test = load_data(FILEPATH_FB_DATA_TEST)
    fb_labels_test = load_data(FILEPATH_FB_LABELS_TEST)

    idx2w, w2idx = data.load_vocab_dict(FILEPATH_VOCAB)
    train_data = data.preprocess(train_data, w2idx)
    fb_data_train = data.preprocess(fb_data_train, w2idx)
    fb_data_test = data.preprocess(fb_data_test, w2idx)

    train_dataset = data.RecSysDataset(X=train_data, y=lab_ratings)
    train_dataloader = train_dataset.create_dataloader(batch_size=64, shuffle=True)

    fb_train_dataset = data.RecSysDataset(X=fb_data_train, y=fb_labels_train)
    fb_train_dataloader = fb_train_dataset.create_dataloader(batch_size=64, shuffle=True)

    fb_test_dataset = data.RecSysDataset(X=fb_data_test, y=fb_labels_test)
    fb_test_dataloader = fb_test_dataset.create_dataloader(batch_size=64, shuffle=False)

    server = models.RecSysServer()
    client = models.RecSysClient(embed_file=EMBED_FILE)

    criterion = nn.MSELoss()
    lr = 0.001
    optimizer_server = Adam(server.parameters(), lr=lr)

    clientsoclist = []
    train_total_batch = []

    total_sendsize_list = []
    total_receivesize_list = []

    client_sendsize_list = [[] for i in range(users)]
    client_receivesize_list = [[] for i in range(users)]

    train_sendsize_list = []
    train_receivesize_list = []

    host = socket.gethostbyname(socket.gethostname())
    port = 10080
    print(host)

    s = socket.socket()
    s.bind((host, port))
    s.listen(5)

    for i in range(users):
        conn, addr = s.accept()
        print('Conntected with', addr)
        clientsoclist.append(conn)  # append client socket on list

        datasize = utils.send_msg(conn, EPOCHS)  # send epoch
        total_sendsize_list.append(datasize)
        client_sendsize_list[i].append(datasize)

        total_batch, datasize = utils.recv_msg(conn)  # get total_batch of train dataset
        total_receivesize_list.append(datasize)
        client_receivesize_list[i].append(datasize)

        train_total_batch.append(total_batch)  # append on list


    trainer = train.TrainerServer(
        server=server,
        client=client,
        device=device,
        loss_fn=criterion,
        optimizer=optimizer_server,
        train_total_batch=train_total_batch,
        clientsoclist=clientsoclist,
        total_receivesize_list=total_receivesize_list,
        client_receivesize_list=client_receivesize_list,
        train_receivesize_list=train_receivesize_list,
    )

    #  train initial dataset
    server, client_weights = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=train_dataloader,
        num_epochs=EPOCHS,
        num_users=users,
        total_sendsize_list=total_sendsize_list,
        client_sendsize_list=client_sendsize_list,
        train_sendsize_list=train_sendsize_list,
    )
    client.load_state_dict(client_weights)

    #  train feedback dataset

    train_total_batch = []

    total_sendsize_list = []
    total_receivesize_list = []

    client_sendsize_list = [[] for i in range(users)]
    client_receivesize_list = [[] for i in range(users)]

    train_sendsize_list = []
    train_receivesize_list = []

    for i in range(users):
        datasize = utils.send_msg(clientsoclist[i], EPOCHS)  # send epoch
        total_sendsize_list.append(datasize)
        client_sendsize_list[i].append(datasize)

        total_batch, datasize = utils.recv_msg(clientsoclist[i])  # get total_batch of train dataset
        total_receivesize_list.append(datasize)
        client_receivesize_list[i].append(datasize)

        train_total_batch.append(total_batch)  # append on list

    trainer = train.TrainerServer(
        server=server,
        client=client,
        device=device,
        loss_fn=criterion,
        optimizer=optimizer_server,
        train_total_batch=train_total_batch,
        clientsoclist=clientsoclist,
        total_receivesize_list=total_receivesize_list,
        client_receivesize_list=client_receivesize_list,
        train_receivesize_list=train_receivesize_list,
    )

    server, client_weights = trainer.train(
        train_dataloader=fb_train_dataloader,
        val_dataloader=fb_test_dataloader,
        num_epochs=EPOCHS,
        num_users=users,
        total_sendsize_list=total_sendsize_list,
        client_sendsize_list=client_sendsize_list,
        train_sendsize_list=train_sendsize_list,
    )


def train_client(users, user_oder):
    device = "cpu"

    train_data = load_data(FILEPATH_DATA)
    lab_ratings = load_data(FILEPATH_RATINGS)
    fb_data_train = load_data(FILEPATH_FB_DATA_TRAIN)
    fb_labels_train = load_data(FILEPATH_FB_LABELS_TRAIN)
    fb_data_test = load_data(FILEPATH_FB_DATA_TEST)
    fb_labels_test = load_data(FILEPATH_FB_LABELS_TEST)

    num_traindata = len(fb_labels_train) // users

    idx2w, w2idx = data.load_vocab_dict(FILEPATH_VOCAB)
    train_data = data.preprocess(train_data, w2idx)
    fb_data_train = data.preprocess(fb_data_train, w2idx, num_traindata, user_oder)
    fb_data_test = data.preprocess(fb_data_test, w2idx)

    fb_labels_train = fb_labels_train[num_traindata * user_oder: num_traindata * (user_oder + 1)]

    train_dataset = data.RecSysDataset(X=train_data, y=lab_ratings)
    train_dataloader = train_dataset.create_dataloader(batch_size=64, shuffle=True)

    fb_train_dataset = data.RecSysDataset(X=fb_data_train, y=fb_labels_train)
    fb_train_dataloader = fb_train_dataset.create_dataloader(batch_size=64, shuffle=True)

    fb_test_dataset = data.RecSysDataset(X=fb_data_test, y=fb_labels_test)
    fb_test_dataloader = fb_test_dataset.create_dataloader(batch_size=64, shuffle=False)

    client = models.RecSysClient(embed_file=EMBED_FILE)

    lr = 0.001
    optimizer_client = Adam(client.parameters(), lr=lr)

    host = '53.73.175.21'
    port = 10080
    s = socket.socket()
    s.connect((host, port))

    total_batch = len(train_dataloader)
    epochs = utils.recv_msg(s)  # get epoch
    msg = total_batch
    utils.send_msg(s, msg)

    train.train_step_client(
        client=client,
        num_epochs=epochs,
        soc=s,
        dataloader=fb_train_dataloader,
        device=device,
        optimizer=optimizer_client,
    )

    total_batch = len(fb_train_dataloader)
    epochs = utils.recv_msg(s)  # get epoch
    msg = total_batch
    utils.send_msg(s, msg)

    train.train_step_client(
        client=client,
        num_epochs=epochs,
        soc=s,
        dataloader=fb_train_dataloader,
        device=device,
        optimizer=optimizer_client,
    )











