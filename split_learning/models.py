import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecSysServer(nn.Module):
    def __init__(self):
        super().__init__()
        #             self.fc1 = nn.Linear(52, 1024)
        #             self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 25)

    def forward(self, x):
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class RecSysClient(nn.Module):
    def __init__(self, embed_file):
        super().__init__()

        embedding_matrix = load_embeddings(embed_file)
        W = torch.Tensor(embedding_matrix)
        self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
        self.embed.weight.data = W.clone()

        self.fc1 = nn.Linear(152, 1024)
        self.fc2 = nn.Linear(1024, 512)

    #         self.fc3 = nn.Linear(512, 256)
    #         self.fc4 = nn.Linear(256, 25)
    def forward(self, age, weight, icd_codes, disease):
        #         age = x['age']
        #         weight = x['weight']
        #         icd_codes = x['icd_codes']

        embedded = self.embed(disease)
        embedded = torch.mean(embedded, 1)

        x = torch.cat((embedded, icd_codes, age, weight), 1).float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


def load_embeddings(embed_file):
    # also normalizes the embeddings
    W = []
    with open(embed_file) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
        # UNK embedding, gaussian randomly initialized
        print("adding unk embedding")
        vec = np.random.randn(len(W[-1]))
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        W.append(vec)
    W = np.array(W)
    return W
