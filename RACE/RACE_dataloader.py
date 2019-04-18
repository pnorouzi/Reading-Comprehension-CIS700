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

from RACE_transforms import RACE_GloVe_Transform

'''
HOW TO USE:
$ from RACE_dataloder import get_dataloaders(transforms = None, batch_size = 32)
This returns train, dev, test dataloaders with appropriate transfroms on the RACE dataset

'''


class RACE_Dataset(data.Dataset):
    def __init__(self, root = 'RACE_Data/', split = 0, transformer = None):
        """ Intializie a RACE Dataset

        :param root: Dataset folder path
        :param split: 0-1-2 for train-dev-test dataset
        :param transforms: pytorch input transform
        :param glove: GloVe emdedding dictionary
        :return RACE_Dataset object
        """
        self.split = split
        self.transformer = transformer

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
        questions = df['questions']
        options = literal_eval(df['options'])
        answer = options[ord(df['answers']) - 65]

        # Perform data transfrom
        if self.transformer:
            article = self.transformer.transform(article)
            question = self.transformer.transform(questions)
            options = [self.transformer.transform(option) for option in options]
            answer = self.transformer.transform(answer)

        return article, question, answer, options


def get_dataloaders(batch_size = 32, transformer_type = 'embded_text_and_pad'):
    """Creates RACE train, dev and test dataloaders

    :param split: int 0 to 2 representing train, dev and test repesctively
    :param batch_size: size of dataloader batch
    :parm glove_embed: bool that specifies if you want your strings to be glove embdded
    :return train, dev, test Pytorch dataloaders
    """

    transformer = RACE_GloVe_Transform(transformer_type = transformer_type)

    train_dataset = RACE_Dataset(split = 0, transformer = transformer)
    dev_dataset = RACE_Dataset(split = 1, transformer = transformer)
    test_dataset = RACE_Dataset(split = 2, transformer = transformer)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader = data.DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)
    test_loader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, dev_loader, test_loader


if __name__ == '__main__':

    transformer = RACE_GloVe_Transform(transformer_type = 'embded_text_and_pad')
    train_loader, dev_loader, test_loader = get_dataloaders(transformer = transformer)

    for (article, questions, answer, options) in train_loader:

        break
