### UNCOMMENT CODE IF RUNNING ON COLAB ###
'''


from colab_logger import Logger

def create_new_logger_instance():
    # TENSOR BOARD SETUP FOR GOOGLE COLAB
    LOG_DIR = './logs'
    get_ipython().system_raw(
        'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
        .format(LOG_DIR)
    )

    !if [ -f ngrok ] ; then echo "Ngrok already installed" ; else wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip > /dev/null 2>&1 && unzip ngrok-stable-linux-amd64.zip > /dev/null 2>&1 ; fi
    get_ipython().system_raw('./ngrok http 6006 &')
    !curl -s http://localhost:4040/api/tunnels | python3 -c \
        "import sys, json; print('Tensorboard Link: ' +str(json.load(sys.stdin)['tunnels'][0]['public_url']))"
logger = create_new_logger_instance()

'''

import numpy as np
import pandas as pd
import pickle
import os

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from dataloader import RACE_Dataset, get_dataloaders
from model import LogReg

# Open preprocessed glove embeddings
file = open("GloVe/processed_glove/word_to_idx_dictionary.pickle",'rb')
word_to_idx= pickle.load(file)
file.close()
file = open("GloVe/processed_glove/weights_matrix.pickle",'rb')
weights_matrix = pickle.load(file)
file.close()

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get Dataloaders
train_loader, dev_loader, test_loader = get_dataloaders(weights_matrix, word_to_idx, batch_size = 32, max_length = 500)

# Initalize model
model = LogReg().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# # Tensorboard connection
# from datetime import datetime
# now = datetime.now()
# logger = Logger('./logs/' + now.strftime("%Y %m %d-%H %M %S") + "/")
# print('Tensorboard model name: ' + now.strftime("%Y %m %d-%H %M %S"))

# Train the model
num_epochs = 10
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):

        x = x.to(device, dtype = torch.float32)
        y = y.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy
        _, y_pred = torch.max(outputs, 1)
        accuracy = (y == y_pred.squeeze()).float().mean()

        # Print to tensorboard
        # info = { 'loss': loss.item(), 'accuracy': accuracy.item() }
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, epoch * len(train_loader) + i)

        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy.item()))

# Dev loader evaluation
accuracy = 0
for i, (x, y) in enumerate(dev_loader):
    x = x.to(device, dtype = torch.float32)
    y = y.to(device)
    # Forward pass
    outputs = model(x)
    _, y_pred = torch.max(outputs, 1)
    accuracy += (y == y_pred.squeeze()).float().sum().item()
print('Accuracy of dev set {}'.format(accuracy/len(dev_loader.dataset)))
