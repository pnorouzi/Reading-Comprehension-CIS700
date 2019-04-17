import pandas as pd
import os

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils import data
from ast import literal_eval

def process_squad(file):
    data = pd.read_json(file)
    contexts = []
    questions = []
    answers_text = []
    answers_start = []
    for i in range(data.shape[0]):
        topic = data.iloc[i,0]['paragraphs']
        for sub_para in topic:
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                answers_start.append(q_a['answers'][0]['answer_start'])
                answers_text.append(q_a['answers'][0]['text'])
                contexts.append(sub_para['context'])
    df = pd.DataFrame({"context":contexts, "question": questions, "answer_start": answers_start, "text": answers_text})
    return df

class SQuAD_Dataset(data.Dataset):
    def __init__(self, root = 'SQuAD_data/', split = 0, transforms = None):
        # root (str): Dataset folder path
        # split (int): 0-1-2 for train-dev-test dataset
        # transforms (transform): pytorch input transform

        self.split = split
        self.transforms = transforms

        self.files = []
        paths = ['train-v1.1.json', 'dev-v1.1.json']
        path = root + paths[split]
        self.files.append(path)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Input: index(int)
        # Output: Paragraph (str), questions (list[str]), answer (str), options (list[str])

        # Open file at index
        file = self.files[index]
        df = process_squad(file)
        # Extract data
        article = df['context'].iloc[index]
        questions = df['question'].iloc[index]
        answer = df['text'].iloc[index]
        # Perform data transfrom if avaliable
        if self.transforms:
            article = self.transforms(article)
            questions = self.transforms(questions)
            answer = self.transforms(answer)

        return article, questions, answer

def get_dataloaders(transforms = None, batch_size = 32):
    # Input: split (int) = 0 for train, 1 for dev, 2 for test
    # Ouput: train, dev, test dataloaders

    train_dataset = SQuAD_Dataset(split = 0, transforms = transforms)
    dev_dataset = SQuAD_Dataset(split = 1, transforms = transforms)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader = data.DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)

    return train_loader, dev_loader

# if __name__ == '__main__':
#     train, _ = get_dataloaders()
#     for x,y,z in train:
#         print(x[0])
#         print(y[0])
#         print(z[0])
#         #print(w[0])
#         break




# if __name__=='__main__':
#     train_data = process_squad("SQuAD_data/train-v1.1.json")
#     valid_data = process_squad("SQuAD_data/dev-v1.1.json")
#     print(train_data.head(5))
