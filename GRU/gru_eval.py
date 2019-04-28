from os.path import exists
import torch
import os
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

from datetime import datetime

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import csv

def alter_file(f):
    with open(f + '.csv') as csv_file:
        with open(f + '_out.csv', 'w') as out_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            writer = csv.writer(out_file)

            line_count = 0
            last = ['article', 'question', 'a', 'b', 'c', 'd', 'y']
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                elif line_count % 4 != 1:
                    idx = (line_count - 1) % 4
                    last[idx+2] = row[2]
                    if (row[3] == '1'):
                        last[6] = str(idx)
                    line_count += 1
                else:
                    writer.writerow(last)
                    last[0] = row[0]
                    last[1] = row[1]
                    last[2] = row[2]
                    if (row[3] == '1'):
                        last[6] = '0'
                    line_count += 1

    assert(line_count % 4 == 1)

class RACE_Dataset_Logistic_Regression(data.Dataset):
    def __init__(self, path, dictionary, weights_matrix, in_idx = True):
        self.dictionary = dictionary
        self.in_idx = in_idx
        self.weights_matrix = weights_matrix
        self.data = pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]

    def __emb__(self, text, limit):
        text_np = self.text_to_word_idx(text)
        text_emb = self.idx_array_to_embdedding_avg(text_np)
        if (text_emb.shape[0] < limit):
            temp = np.zeros((limit,300))
            temp[:text_emb.shape[0],:] = text_emb
            text_emb = temp
        else:
            text_emb = text_emb[-limit:,:]
        return torch.from_numpy(text_emb)

    def __getitem__(self, index):
        df = self.data.iloc[index]
        article = self.__emb__(df['article'], 400)
        question = self.__emb__(df['question'], 30)
        a = self.__emb__(df['a'], 16)
        b = self.__emb__(df['b'], 16)
        c = self.__emb__(df['c'], 16)
        d = self.__emb__(df['d'], 16)
        return article, question, a, b, c, d, torch.tensor(int(df['y']))

    def text_to_word_idx(self, text):
        test_idx = []
        for word in text.split(' '):
            test_idx.append(self.dictionary[word])
        return np.array(test_idx)

    def idx_array_to_embdedding_avg(self, data):
        all_embeddings = np.zeros((len(data),300))
        for i, num in enumerate(data):
            all_embeddings[i, :] = self.weights_matrix[num,:]
        return all_embeddings

def get_dataloaders(batch_size = 32):

    with open('data/word_to_idx_dictionary.pickle', 'rb') as handle:
        dictionary = pickle.load(handle)
    with open('data/weights_matrix.pickle', 'rb') as handle:
        weights_matrix = pickle.load(handle)

    train_dataset = RACE_Dataset_Logistic_Regression('data/train_data_out.csv',  dictionary, weights_matrix, in_idx = True)
    dev_dataset   = RACE_Dataset_Logistic_Regression('data/dev_data_out.csv',   dictionary, weights_matrix, in_idx = True)
    test_dataset  = RACE_Dataset_Logistic_Regression('data/test_data_out.csv',  dictionary, weights_matrix, in_idx = True)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader   = data.DataLoader(dev_dataset,   batch_size = batch_size, shuffle = True)
    test_loader  = data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = True)

    return train_loader, dev_loader, test_loader

class GRU_Network(nn.Module):
    def __init__(self):
        super(GRU_Network, self).__init__()

        self.art = nn.GRU(300, 64, batch_first=True, bidirectional=True)

        self.que = nn.GRU(300, 32, batch_first=True, bidirectional=True)

        self.opt = nn.GRU(300, 32, batch_first=True, bidirectional=True)

        self.final = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(2*(64 + 32*5), 16),
            nn.LeakyReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

    def forward(self, article, q, a, b, c, d):
        article, hart = self.art(article)
        q, hq = self.que(q)
        a, ha = self.opt(a)
        b, hb = self.opt(b)
        c, hc = self.opt(c)
        d, hd = self.opt(d)

        full = torch.cat((article[:,-1,:], q[:,-1,:]), dim=1)
        full = torch.cat((full, a[:,-1,:]), dim=1)
        full = torch.cat((full, b[:,-1,:]), dim=1)
        full = torch.cat((full, c[:,-1,:]), dim=1)
        full = torch.cat((full, d[:,-1,:]), dim=1)
        return self.final(full)

def tests(mdl, dev_loader, test_loader):

    # DEV TEST
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (art, que, a, b, c, d, y) in enumerate(dev_loader):

            art = art.to(device, dtype = torch.float32)
            que = que.to(device, dtype = torch.float32)
            a = a.to(device, dtype = torch.float32)
            b = b.to(device, dtype = torch.float32)
            c = c.to(device, dtype = torch.float32)
            d = d.to(device, dtype = torch.float32)
            y = y.to(device)

            yhat = mdl.forward(art, que, a, b, c, d)
            _, y_pred = torch.max(yhat, 1)
            correct += (y == y_pred).sum().item()
            total += y.shape[0]
        print('Accuracy of the network on the dev set: {} %'.format(100 * correct / total))

    with torch.no_grad():
        correct = 0
        total = 0
        for i, (art, que, a, b, c, d, y) in enumerate(test_loader):

            art = art.to(device, dtype = torch.float32)
            que = que.to(device, dtype = torch.float32)
            a = a.to(device, dtype = torch.float32)
            b = b.to(device, dtype = torch.float32)
            c = c.to(device, dtype = torch.float32)
            d = d.to(device, dtype = torch.float32)
            y = y.to(device)

            yhat = mdl.forward(art, que, a, b, c, d)
            _, y_pred = torch.max(yhat, 1)
            correct += (y == y_pred).sum().item()
            total += y.shape[0]
        print('Accuracy of the network on the test set: {} %'.format(100 * correct / total))


files = ['data/train_data', 'data/dev_data', 'data/test_data']

for f in files:
    alter_file(f)

train_loader, dev_loader, test_loader = get_dataloaders(batch_size = 32)

gru_mdl = GRU_Network().to(device)
gru_mdl.load_state_dict(torch.load('gru_mdl.pt'))
tests(gru_mdl, dev_loader, test_loader)
