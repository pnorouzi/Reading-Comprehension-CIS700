import os
import pickle
import bcolz
import numpy as np

''' Pre Processing of GloVe word embeddings 
This script will parse GloVe dataset from https://nlp.stanford.edu/projects/glove/
You need to download whichever representation you want in this working directory
and unzip it and specify which size latent representation you want 50-100-200-300
default is 300. It will create a pickle in the correct format that we are using 
in he rest of the project. Below in comments is how you correclty open the word
embedding glove dictionary
Also, thanks to https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
for the code :) 
'''

# Size of representation (check that it's in the dataset)
latent_size = 300

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.{latent_size}.dat', mode='w')

with open(f'{glove_path}/glove.6B.{latent_size}d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 300)), rootdir=f'{glove_path}/6B.{latent_size}.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.{latent_size}_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.{latent_size}_idx.pkl', 'wb'))


'''
vectors = bcolz.open(f'{glove_path}/6B.{latent_size}.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.{latent_size}_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.{latent_size}_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}
'''