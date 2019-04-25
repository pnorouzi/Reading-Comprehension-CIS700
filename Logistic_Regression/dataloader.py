import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset


class RACE_Dataset(data.Dataset):
    def __init__(self, path, weights_matrix, word_to_idx, max_length = 500):
        self.max_length = max_length
        self.data = pd.read_csv(path)
        self.weights_matrix = weights_matrix
        self.word_to_idx = word_to_idx

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        # Extract data from dataframe
        article  = data_row['article']
        question = data_row['questions']
        answer_index  = int(data_row['answers'])
        option1  = data_row['option1']
        option2  = data_row['option2']
        option3  = data_row['option3']
        option4  = data_row['option4']

        # Convert text to ids
        article = self.convert_text_to_ids(article)
        question = self.convert_text_to_ids(question)
        option1 = self.convert_text_to_ids(option1)
        option2 = self.convert_text_to_ids(option2)
        option3 = self.convert_text_to_ids(option3)
        option4 = self.convert_text_to_ids(option4)

        # Pad to max_length
        article = self.pad_list(article, sequence_length = self.max_length)
        question= self.pad_list(question, sequence_length = self.max_length)
        option1 = self.pad_list(option1, sequence_length = self.max_length)
        option2 = self.pad_list(option2, sequence_length = self.max_length)
        option3 = self.pad_list(option3, sequence_length = self.max_length)
        option4 = self.pad_list(option4, sequence_length = self.max_length)

        # Embded using glove weights matrix
        article = self.embed(article)
        question = self.embed(question)
        option1 = self.embed(option1)
        option2 = self.embed(option2)
        option3 = self.embed(option3)
        option4 = self.embed(option4)

        # Convert to tensors
        article = torch.from_numpy(article)
        question = torch.from_numpy(question)
        option1 = torch.from_numpy(option1)
        option2 = torch.from_numpy(option2)
        option3 = torch.from_numpy(option3)
        option4 = torch.from_numpy(option4)

        # Concat each article, question, option
        concat_1 = torch.cat([article, question, option1], 0)
        concat_2 = torch.cat([article, question, option2], 0)
        concat_3 = torch.cat([article, question, option3], 0)
        concat_4 = torch.cat([article, question, option4], 0)

        # Concat each of the above concats into a 4x1500 tensors
        output = torch.stack([concat_1, concat_2, concat_3, concat_4], 0)

        return output, torch.tensor(answer_index)

    def convert_text_to_ids(self, text):
        # Do some data cleaning to correclty
        # tokenize the whole text
        text = text.replace('.', ' . ')
        text = text.replace(',', ' , ')
        text = text.replace('!', ' ! ')
        text = text.replace('?', ' ? ')
        text = text.replace('$', ' $ ')
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace(':', ' : ')
        text = text.replace('"', ' " ')

        # Use word_to_idx dict to find word
        # index in weights_matrix
        ids = []
        for word in text.split(' '):
            if word in self.word_to_idx and len(word) > 0:
                ids.append(self.word_to_idx[word])
            else:
                ids.append(0)
        return ids

    def pad_list(self, input_list, sequence_length = 512, fillvalue = 0):
        output = input_list + [fillvalue] * (sequence_length - len(input_list))
        if len(output) > sequence_length:
            output = output[-sequence_length:]
        return output

    def embed(self, ids):
        embedding = np.zeros((self.max_length, 300))
        for i, idx in enumerate(ids):
            embedding[i, :] = self.weights_matrix[idx]
        return embedding.mean(axis = 0)



def get_dataloaders(weights_matrix, word_to_idx, batch_size = 32, max_length = 500):

    # Initalize Datasets
    train_dataset = RACE_Dataset('Data/train_data.csv', weights_matrix, word_to_idx, max_length = max_length)
    dev_dataset   = RACE_Dataset('Data/dev_data.csv',weights_matrix, word_to_idx, max_length = max_length)
    test_dataset  = RACE_Dataset('Data/test_data.csv', weights_matrix, word_to_idx,max_length = max_length)

    # Load data loaders
    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader   = data.DataLoader(dev_dataset,   batch_size = batch_size, shuffle = True)
    test_loader  = data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = True)

    return train_loader, dev_loader, test_loader

# How to use?
#train_loader, dev_loader, test_loader = get_dataloaders(weights_matrix, word_to_idx, batch_size = 32, max_length = 500)
