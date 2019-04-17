import os

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils import data
from ast import literal_eval

class RACE_Dataset(data.Dataset):
    def __init__(self, root = 'RACE_Data/', split = 0, transforms = None):
        # root (str): Dataset folder path
        # split (int): 0-1-2 for train-dev-test dataset
        # transforms (transform): pytorch input transform

        self.split = split
        self.transforms = transforms

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
        # Output: Paragraph (str), questions (list[str]), answer (str), options (list[str])

        # Open file at index
        file = self.files[index]
        df = pd.read_csv(file).iloc[0]

        # Extract text answer from list of options
        letter_to_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
        # print(df['answers'])
        # print(letter_to_index[df['answers']])
        # print(literal_eval(df['options']))
        answer = literal_eval(df['options'])[letter_to_index[df['answers']]]

        # Extract data
        article = df['article']
        questions = df['questions']
        options = df['options']

        # Perform data transfrom if avaliable
        if self.transforms:
            article = self.transforms(article)
            questions = self.transforms(questions)
            options = self.transforms(options)
            answer = self.transforms(answer)

        return article, questions, answer, options

def get_dataloaders(transforms = None, batch_size = 32):
    # Input: split (int) = 0 for train, 1 for dev, 2 for test
    # Ouput: train, dev, test dataloaders

    train_dataset = RACE_Dataset(split = 0, transforms = transforms)
    dev_dataset = RACE_Dataset(split = 1, transforms = transforms)
    test_dataset = RACE_Dataset(split = 2, transforms = transforms)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader = data.DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)
    test_loader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, dev_loader, test_loader

# if __name__ == '__main__':
#     train, _, _ = get_dataloaders()
#     for x,y,z,w in train:
#         print(x[0])
#         print(y[0])
#         print(z[0])
#         #print(w[0])
#         break
