import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

#from dataloader import RACE_Dataset, get_dataloaders
# from model import CNN
import pickle

file = open("word_to_idx_dictionary.pickle",'rb')
word_to_idx= pickle.load(file)
file.close()
file = open("weights_matrix.pickle",'rb')
weights_matrix = pickle.load(file)
file.close()
#len(word_to_idx)
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.article_layer = nn.Sequential(
            nn.Conv1d(300,64,3, padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),     
            nn.Conv1d(64,32,3, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
        )
        
        self.question_layer = nn.Sequential(
            nn.Conv1d(300,64,3, padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),     
            nn.Conv1d(64,32,3, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
        )
       
        self.options_layer = nn.Sequential(
            nn.Conv1d(300,64,3, padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),     
            nn.Conv1d(64,32,3, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
        )
        
        self.classifier_1 = nn.Linear(48000, 5024)
        self.classifier_2 = nn.Linear(5024,1)
        
    def forward(self, article, question, options):
        out_article = self.article_layer(article)
        out_article = out_article.view(out_article.shape[0], -1)
        out_question = self.question_layer(question)
        out_question = out_question.view(out_question.shape[0], -1)

        opt_1 = options[:,0,:,:]
        opt_2 = options[:,1,:,:]
        opt_3 = options[:,2,:,:]
        opt_4 = options[:,3,:,:]
        out_opt_1 = self.options_layer(opt_1)
        out_opt_2 = self.options_layer(opt_2)
        out_opt_3 = self.options_layer(opt_3)
        out_opt_4 = self.options_layer(opt_4)
        
        out_opt_1 = out_opt_1.view(out_opt_1.shape[0], -1)
        out_opt_2 = out_opt_2.view(out_opt_2.shape[0], -1)
        out_opt_3 = out_opt_3.view(out_opt_3.shape[0], -1)
        out_opt_4 = out_opt_4.view(out_opt_4.shape[0], -1)

        
        # Conact stuff
        opt_1 = torch.cat([out_article,out_question, out_opt_1],1)
        opt_2 = torch.cat([out_article,out_question, out_opt_2],1)
        opt_3 = torch.cat([out_article,out_question, out_opt_3],1)
        opt_4 = torch.cat([out_article,out_question, out_opt_4],1)

        
        x = torch.stack([opt_1, opt_2, opt_3, opt_4], 1)
        
        x = self.classifier_1(x)
        logits = self.classifier_2(x)
        reshaped_logits = logits.view(-1, 4)
        return reshaped_logits
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
        question= self.pad_list(question, sequence_length = 50)
        option1 = self.pad_list(option1, sequence_length = 25)
        option2 = self.pad_list(option2, sequence_length = 25)
        option3 = self.pad_list(option3, sequence_length = 25)
        option4 = self.pad_list(option4, sequence_length = 25)

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

        # Concat each of the above concats into a 4x1500 tensors
        options = torch.stack([option1, option2, option3, option4],0)

        return article, question, options, torch.tensor(answer_index)

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
        embedding = np.zeros((300, self.max_length))
        for i, idx in enumerate(ids):
            embedding[:, i] = self.weights_matrix[idx]

        return embedding



def get_dataloaders(weights_matrix, word_to_idx, batch_size = 32, max_length = 500):

    train_dataset = RACE_Dataset('Data/train_data.csv', weights_matrix, word_to_idx, max_length = max_length)
    dev_dataset   = RACE_Dataset('Data/dev_data.csv',weights_matrix, word_to_idx, max_length = max_length)
    test_dataset  = RACE_Dataset('Data/test_data.csv', weights_matrix, word_to_idx,max_length = max_length)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    dev_loader   = data.DataLoader(dev_dataset,   batch_size = batch_size, shuffle = True)
    test_loader  = data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = False)

    return train_loader, dev_loader, test_loader


train_loader, dev_loader, test_loader = get_dataloaders(weights_matrix, word_to_idx, batch_size = 32, max_length = 500)
file = open('cnn_model_5epochs.pickle','rb')
model = pickle.load(file)
file.close()

#model = CNN()
#print(model)
#loaded = torch.load("logistic_regression.pt",map_location='cpu')
#print(loaded)

#model.load_state_dict(loaded)

#train_loader, dev_loader, test_loader = get_dataloaders(weights_matrix, word_to_idx, batch_size = 32, max_length = 500)

device = 'cuda:0'

accuracy = 0
for i, (article, question, options, y) in enumerate(test_loader):


    #x = x.type(torch.float32)
    y = y.to(device)
    # Forward pass
    outputs = model(article.to(device, dtype = torch.float32), question.to(device, dtype = torch.float32), options.to(device, dtype = torch.float32))
    _, y_pred = torch.max(outputs, 1)
    #print('predicted',y_pred)
    #print('Actual',y)
    accuracy += (y == y_pred.squeeze()).float().sum().item()
print('Accuracy of dev set {}'.format(accuracy/len(test_loader.dataset)))
