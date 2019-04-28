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
LOG_DIR = './logs'
'''
# TENSOR BOARD SETUP FOR GOOGLE COLAB

get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

!if [ -f ngrok ] ; then echo "Ngrok already installed" ; else wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip > /dev/null 2>&1 && unzip ngrok-stable-linux-amd64.zip > /dev/null 2>&1 ; fi
get_ipython().system_raw('./ngrok http 6006 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print('Tensorboard Link: ' +str(json.load(sys.stdin)['tunnels'][0]['public_url']))"

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
'''
import tensorflow as tf
import numpy as np
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

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


def train_model(mdl, bs, learning_rate, num_epochs, text):
    train_loader, dev_loader, test_loader = get_dataloaders(batch_size = bs)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=learning_rate)

    # Tensorboard connection
    now = datetime.now()
    logger = Logger('./logs/' + now.strftime("%Y %m %d-%H %M %S") + "/")
    print('Tensorboard model name: ' + now.strftime("%Y %m %d-%H %M %S"))

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (art, que, a, b, c, d, y) in enumerate(train_loader):

            art = art.to(device, dtype = torch.float32)
            que = que.to(device, dtype = torch.float32)
            a = a.to(device, dtype = torch.float32)
            b = b.to(device, dtype = torch.float32)
            c = c.to(device, dtype = torch.float32)
            d = d.to(device, dtype = torch.float32)
            y = y.to(device)

            # Forward pass
            yhat = mdl.forward(art, que, a, b, c, d)
            loss = criterion(yhat, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, y_pred = torch.max(yhat, 1)
            accuracy = (y == y_pred.squeeze()).float().mean()

            info = { str(text + 'Loss'): loss.item(), str(text + 'Accuracy'): accuracy.item() }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch * len(train_loader) + i)

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                print(accuracy.item())

    return mdl, dev_loader, test_loader

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

gru_mdl, dev_loader, test_loader = train_model(GRU_Network().to(device), 32, 0.002, 3, "GRU ")
tests(gru_mdl, dev_loader, test_loader)
