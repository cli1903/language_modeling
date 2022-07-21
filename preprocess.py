import tensorflow as tf
import numpy as np
from functools import reduce

def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    # TODO: load and concatenate training data from training file.

    # TODO: load and concatenate testing data from testing file.

    # TODO: read in and tokenize training data

    # TODO: read in and tokenize testing data

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.

    with open(train_file) as train:
        with open(test_file) as test:
            train_words = train.read().split()
            test_words = test.read().split()
            
            train_arr = []
            test_arr = []

            vocab_dict = {}
            id_count = 0
            
            for w in train_words:
                if w not in vocab_dict:
                    vocab_dict[w] = id_count
                    train_arr.append(id_count)
                    id_count += 1
                else:
                    train_arr.append(vocab_dict[w])
                    
            for w in test_words:
                if w not in vocab_dict:
                    vocab_dict[w] = id_count
                    test_arr.append(id_count)
                    id_count += 1
                else:
                    test_arr.append(vocab_dict[w])
                    
            return train_arr, test_arr, vocab_dict
                    