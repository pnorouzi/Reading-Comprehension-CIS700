import os
import pickle
import bcolz
import numpy as np
import torch

''' RACE_GloVe_Transform  
This class converts string text into matricies of word vectors.
If converts each word to a vector of specified size and then concatenates all 
words in text along dimension 1.

How to use?

$ transformer = GloVe_Transform()
$ string = ...some string...
$ embdedded_string = transformer.emebed_text_and_pad(string, max_length = 1000)

Tranfromer Options?
emebed_text_and_pad(string, max_length = 4000): embeds each word as vector and pads/truncates to max_length with 0s
embded_and_average(string): embeds each word and then takes the average of all the embedded vectors

'''

class GloVe_Transform:
    def __init__(self, glove_size = 300):
        # Load GloVe embedding dictionary
        glove_path = '../GloVe/glove.6B'
        vectors = bcolz.open(f'{glove_path}/6B.{str(glove_size)}.dat')[:]
        words = pickle.load(open(f'{glove_path}/6B.{str(glove_size)}_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.{str(glove_size)}_idx.pkl', 'rb'))
        self.glove = {w: vectors[word2idx[w]] for w in words}
        self.glove_size = glove_size
            
    def preprocess_text(self, string):
        """ String Cleaner
        Takes a string cleans it according to edge cases and returns a list of words

        :param string: input string
        :return list of words
        """

        string = string.replace('.', ' . ')
        string = string.replace('"', ' " ')
        string = string.replace('\n', ' | ')
        string = string.replace('?', ' ? ')
        string = string.replace('!', ' ! ')
        
        return string.split(' ')

    def emebed_and_pad(self, string, max_length = 4000):
        """ GloVe Word Embedding
        Takes in a string however long and converts into word embedded np.array of
        size (glove-size)x(max_length) that is each column is a word. Moroever,
        it pads the with zeros to max_length size. Truncates if too long.

        :param string: input string you want to convert to vector representation
        :param max_length: size of dim 1 in output
        :return torch tensor of size (glove-size)x(max_length)
        """

        words = self.preprocess_text(string)
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

    def embed_and_average(self, string):
        """ GloVe Average Word Embdeding
        Takes in a string and convertes it to embded vectors matrix
        and then takes the average

        :param string: input string
        :return torch tensor of size glove_size
        """

        output = self.emebed_and_pad(string, max_length = int(len(string)/2))

        # Average word vectors
        output = output.numpy().mean(axis = 1)

        return torch.from_numpy(output)

    def none(self, string):
        return string