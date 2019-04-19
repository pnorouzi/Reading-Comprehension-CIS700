import os
import pickle
import bcolz
import numpy as np
from ast import literal_eval

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils import data


class RACE_Dataset(data.Dataset):
    def __init__(self, root = 'RACE_Data/', split = 0):
        """ Intializie a RACE Dataset

        :param root: Dataset folder path
        :param split: 0-1-2 for train-dev-test dataset
        :return RACE_Dataset object
        """
        self.split = split

        # Extract file paths for dataset
        self.files = []
        paths = ['train/', 'dev/', 'test/']
        path = root + paths[split]
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                self.files.append(os.path.join(root,filename))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Input: index(int)
        # Output: Paragraph, question, answer, options (list[])

        # Open file at index
        file = self.files[index]
        df = pd.read_csv(file).iloc[0]

        # Extract data
        article = df['article']
        question = df['questions']
        options = '||OPTION BREAK||'.join(literal_eval(df['options']))
        answer_index = ord(df['answers']) - 65
        answer = literal_eval(df['options'])[answer_index]

        return article, question, answer, options, answer_index


def get_dataloaders(batch_size = 32):
    """Creates RACE train, dev and test dataloaders

    :param batch_size: size of dataloader batch
    :return train, dev, test Pytorch dataloaders
    """

    train_dataset = RACE_Dataset(split = 0)
    dev_dataset   = RACE_Dataset(split = 1)
    test_dataset  = RACE_Dataset(split = 2)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader   = data.DataLoader(dev_dataset,   batch_size = batch_size, shuffle = True)
    test_loader  = data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = True)

    return train_loader, dev_loader, test_loader

# TESTING CODE
# if __name__ == '__main__':
#     train_loader, dev_loader, test_loader = get_dataloaders(transformer_type = 'none')
#     for (article, question, answer, options, answer_index) in train_loader:
#         break
