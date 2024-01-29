
import copy
import time

import torch
import numpy as np
from tqdm import tqdm


from utils import recv_msg, send_msg


class TrainerServer:
    def __init__(
            self,
            server,
            client,
            device,
            loss_fn,
            optimizer,
            train_total_batch,
            clientsoclist,
            total_receivesize_list,
            client_receivesize_list,
            train_receivesize_list,
    ):
        self.server = server
        self.client = client
        self.client_weights = copy.deepcopy(client.state_dict())
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_total_batch = train_total_batch
        self.clientsoclist = clientsoclist
        self.total_receivesize_list = total_receivesize_list
        self.client_receivesize_list = client_receivesize_list
        self.train_receivesize_list = train_receivesize_list

    def train_step(
            self,
            epoch,
            user,
            total_sendsize_list,
            client_sendsize_list,
            train_sendsize_list,
    ):
        for i in tqdm(range(self.train_total_batch[user]), ncols=100, desc='Epoch {} Client{} '.format(epoch + 1, user)):
            self.optimizer.zero_grad()  # initialize all gradients to zero

            msg, datasize = recv_msg(self.clientsoclist[user])  # receive client message from socket
            self.total_receivesize_list.append(datasize)
            self.client_receivesize_list[user].append(datasize)
            self.train_receivesize_list.append(datasize)

            client_output_cpu = msg['client_output']  # client output tensor
            label = msg['label']  # label

            client_output = client_output_cpu.to(self.device)
            label = label.clone().detach().float().to(self.device)

            output = self.server(client_output)  # forward propagation
            loss = self.loss_fn(output, label)  # calculates cross-entropy loss
            loss.backward()  # backward propagation
            msg = client_output_cpu.grad.clone().detach()

            datasize = send_msg(self.clientsoclist[user], msg)
            total_sendsize_list.append(datasize)
            client_sendsize_list[user].append(datasize)
            train_sendsize_list.append(datasize)

            self.optimizer.step()

        self.client_weights, datasize = recv_msg(self.clientsoclist[user])
        self.total_receivesize_list.append(datasize)
        self.client_receivesize_list[user].append(datasize)
        self.train_receivesize_list.append(datasize)

    def eval_step(
            self,
            dataloader,
    ):

        with torch.no_grad():
            loss = 0.0
            pred = torch.empty((0, 25))
            labels = torch.empty((0, 25))
            for j, trn in enumerate(dataloader):
                age, weight, icd_codes, disease = trn['age'], trn['weight'], trn['icd_codes'], trn['disease']
                target = trn['target']

                age = age.to(self.device)
                weight = weight.to(self.device)
                icd_codes = icd_codes.to(self.device)
                disease = disease.to(self.device)
                target = target.to(self.device)

                output = self.client(age, weight, icd_codes, disease)
                output = self.server(output)
                target = target.float()
                loss += self.loss_fn(output, target).item()
                pred = torch.cat((pred, output))
                labels = torch.cat((labels, target))

            loss_avg = loss / len(dataloader)
            pred = pred.numpy()
            labels = labels.numpy()

            prec_k = precision_at_k(labels, pred, 5)

            return loss_avg, prec_k

    def train(
            self,
            train_dataloader,
            val_dataloader,
            num_epochs,
            num_users,
            total_sendsize_list,
            client_sendsize_list,
            train_sendsize_list,
    ):

        for epoch in range(num_epochs):
            for user in range(num_users):
                datasize = send_msg(self.clientsoclist[user], self.client_weights)
                total_sendsize_list.append(datasize)
                client_sendsize_list[user].append(datasize)
                train_sendsize_list.append(datasize)
                self.train_step(
                    epoch=epoch,
                    user=user,
                    total_sendsize_list=total_sendsize_list,
                    client_sendsize_list=client_sendsize_list,
                    train_sendsize_list=train_sendsize_list,
                )

            self.client.load_state_dict(self.client_weights)
            self.client.to(self.device)
            self.client.eval()

            train_loss = self.eval_step(
                dataloader=train_dataloader,
            )
            print(f"train_loss: {train_loss:.4f}")

            val_loss, prec_k = self.eval_step(
                dataloader=val_dataloader,
            )
            print(f"val_loss: {val_loss:.4f}")
            print(f"Precision@k: {prec_k:.4f}")

            return self.server, self.client_weights


def train_step_client(
        client,
        num_epochs,
        soc,
        dataloader,
        device,
        optimizer,
):
    for e in range(num_epochs):
        client_weights = recv_msg(soc)
        client.load_state_dict(client_weights)
        client.eval()
        for i, batch_data in enumerate(tqdm(dataloader, ncols=100, desc='Epoch ' + str(e + 1))):
            age, weight, icd_codes, disease = batch_data['age'], batch_data['weight'], batch_data['icd_codes'], batch_data['disease']
            target = batch_data['target']

            age = age.to(device)
            weight = weight.to(device)
            icd_codes = icd_codes.to(device)
            disease = disease.to(device)

            target = target.to(device)

            optimizer.zero_grad()
            output = client(age, weight, icd_codes, disease)
            client_output = output.clone().detach().requires_grad_(True)
            msg = {
                'client_output': client_output,
                'label': target
            }
            send_msg(soc, msg)
            client_grad = recv_msg(soc)
            output.backward(client_grad)
            optimizer.step()
        send_msg(soc, client.state_dict())
        time.sleep(0.5)


def precision_at_k(y_true, y_pred, k):
    topk_true = np.flip(np.argsort(y_true), 1)[:, :k]
    topk_pred = np.flip(np.argsort(y_pred), 1)[:, :k]

    n_relevant = 0
    n_recommend = 0

    for t, p in zip(topk_true, topk_pred):
        # print(f"t:{t}")
        # print(f"p:{p}")
        n_relevant += len(np.intersect1d(t, p))
        # print(f"rev:{n_relevant}")
        n_recommend += len(p)
        # print(n_recommend)

    return float(n_relevant) / n_recommend
