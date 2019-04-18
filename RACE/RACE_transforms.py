import os
import pickle
import bcolz
import numpy as np
import torch

'''

transformer_types:
:pad
:average

'''

class RACE_GloVe_Transform:
    def __init__(self, transformer_type = 'embded_text_and_pad', glove_size = 300):
        glove_path = '../GloVe/glove.6B'
        vectors = bcolz.open(f'{glove_path}/6B.{str(glove_size)}.dat')[:]
        words = pickle.load(open(f'{glove_path}/6B.{str(glove_size)}_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.{str(glove_size)}_idx.pkl', 'rb'))

        self.glove = {w: vectors[word2idx[w]] for w in words}
        self.glove_size = glove_size

        if transformer_type == 'pad':
            self.transform = self.emebed_text_and_pad
        elif transformer_type == 'average':
            self.transform = self.embed_text_and_average

    def emebed_text_and_pad(self, string, max_length = 4000):
        """ GloVe Word Embedding
        Takes in a string however long and converts into word embedded np.array of
        size (glove-size)x(max_length) that is each column is a word. Moroever,
        it pads the with zeros to max_length size. Truncates if too long.

        :param string: input string you want to convert to vector representation
        :param max_length: size of dim 1 in output
        :return torch tensor of size (glove-size)x(max_length)
        """

        string = string.replace('.', ' . ')
        string = string.replace('"', ' " ')
        string = string.replace('\n', ' | ')
        string = string.replace('?', ' ? ')
        string = string.replace('!', ' ! ')

        words = string.split(' ')
        num_words = len(words)
        glove_size = self.glove_size

        #print(words)
        output = np.zeros((glove_size,max_length))
        for index, word in enumerate(words):
            # Convert word to vector
            if word in self.glove:
                output[:, index] = self.glove[word.lower()]
            else:
                #print(word)
                pass
            # Truncates if there are more words than max_length
            if index == max_length - 1:
                break

        return torch.from_numpy(output)

    def embed_text_and_average(self, string):
        """ GloVe Average Word Embdeding
        Takes in a string and convertes it to embded vectors matrix
        and then takes the average

        :param string: input string
        :return torch tensor of size glove_size
        """

        output = self.emebed_text_and_pad(string, max_length = len(string)/2)

        # Average word vectors
        output = np.mean(output, axis = 1)

        return torch.from_numpy(output)
