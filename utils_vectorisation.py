'''Module providing utility functions for vectorisation of tweets'''

import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

embeddings_directory = os.environ['EMBEDDINGS']
data_directory = os.environ['DATA']


def create_raw_X_Y(dataframe, classification='subjectivity'):
    '''Creates numpy data structures containing the tokens of tweets and corresponding classes for either subjectivity classification or polarity classification
    :param: with columns: 'tokens' (lists of strings), 'polarity' (floats/ints), 'subjectivity' (floats/ints)
    :type: pandas dataframe
    :kparam: classification, the type of classification, either subjecitivity or polarity
    :type: string
    :return: raw_X, list with m variable length lists of strings containing the tokens of the tweets
    :rtype: list of lists of strings
    :return: raw_Y, list of floats/ints either 1 or 0
    '''
    if classification == 'subjectivity':
        df = dataframe[dataframe['subjectivity'].isin([0, 1])]
        raw_X = list(df['tokens'])
        raw_Y = list(df['subjectivity'])
        return raw_X, raw_Y
    elif classification == 'polarity':
        df = dataframe[dataframe['polarity'].isin([0, 1])]
        raw_X = list(df['tokens'])
        raw_Y = list(df['polarity'])
        return raw_X, raw_Y
    else:
        raise ValueError('classification kwarg not recognised.  Must be either \'subjectivity\' or \'polarity\'')


def split_train_test(X, Y, percentage_split=0.9):
    '''Returns train and test data according to % split
    :param: X, Y (full dataset)
    :type: numpy ndarrays, lists or other subcriptable object
    :param: percentage split, number between 0 and 1
    :type: float
    :return: X_train, Y_train, X_test, Y_test
    :rtype: numpy ndarrays
    '''
    cutoff = int(len(X) * percentage_split)
    X_train = X[:cutoff]
    Y_train = Y[:cutoff]
    X_test = X[cutoff:]
    Y_test = Y[cutoff:]
    return (X_train, Y_train), (X_test, Y_test)


def balance_data(raw_X, raw_Y):
    '''Take an unbalanced dataset and returns a balanced, shuffled dataset
    :param: X, matrix of feature vectors for each observation
    :type: list of lists
    :param: Y, vector of classes for each observation corresponding to X
    :type: list
    :return: balanced_shuffled_X, balanced_shuffled_Y
    :rtype: numpy ndarrays
    '''
    # create and populate lists of features and target variable for each class
    pos = []
    neg = []
    for i in range(len(raw_X)):
        if raw_Y[i] == 1:
            pos.append([raw_X[i], raw_Y[i]])
        else:
            neg.append([raw_X[i], raw_Y[i]])

    # shuffle the lists and put together a balanced data_set using the length of the shortest list, discarding the samples from the longer list
    length = min(len(pos), len(neg))
    lst = [pos, neg]
    for index in range(len(lst)):
        np.random.shuffle(lst[index])
    data_set = pos[0:length] + neg[0:length]

    balanced_X = np.array([element[0] for element in data_set])
    balanced_Y = np.array([element[1] for element in data_set])

    # shuffle the dataset again
    perm = list(np.random.permutation(len(balanced_X)))
    balanced_shuffled_X = balanced_X[perm]
    balanced_shuffled_Y = balanced_Y[perm]

    return balanced_shuffled_X, balanced_shuffled_Y


def normalise(X):
    '''Accepts an matrix of m observations of n features and returns a normalised matrix where the features each have a mean of zero and variance 1
    :param: X, dataset of vectorised tweets (the output of bow)
    :type: numpy ndarray
    :return: X_normalised, normalised dataset of vectorised tweets
    :rtype: numpy ndarray
    '''
    return X / X.max(axis=0)


def token_index_map(X):
    '''returns {token: index} dictionary from a list of lists of tokens
    :param: X, list of lists of tokens
    :type: list of lists of strings
    :return: dictionary of string:int pairs
    :rtype: dictionary
    '''
    token_index_map = {}
    count = 0
    for row in X:
        for token in row:
            if token_index_map.get(token, None) == None:
                token_index_map[token] = count
                count += 1

    return token_index_map


def bow(pp_dataframe, target='subjectivity'):
    '''From a dataframe of preprocessed tweets with annotated sentiment returns a normalised bag of words with corresponding classes
    :param: pp_dataframe; dataframe with 'tokens', 'subjectivity', 'polarity' columns
    :type: pandas dataframe, 'tokens': list of strings, subjectivity and polarity are ints
    :param: classification; either 'subjectivity' or 'polarity'
    :type: string
    :return: X; m x n matrix of feature vectors, m = number of observations, n = feature vector size (size of BoW word index)
    :rtype: numpy ndarray
    :return: Y; vector of classes
    :rtype: numpy ndarray
    :return: word_index; the word index corresponding to the BoW
    :rtype: dictionary
    '''
    # use functions to get a balanced preprocessed token matrix and list of target vectors
    raw_X, raw_Y = create_raw_X_Y(pp_dataframe, classification=target)
    X, Y = balance_data(raw_X, raw_Y)
    # create token index map to store word corresponding to index of bow vector
    timap = token_index_map(X)
    num_tokens = max([timap[key] for key in timap]) + 1
    vectors = []
    for i in range(len(X)):
        vector = [0 for index in range(num_tokens)]
        for token in X[i]:
            vector[timap[token]] += 1
        vectors.append(vector)

    vectors = np.array(vectors)
    vectors = normalise(vectors)
    Y = np.array(Y)

    return vectors, Y


def sequence_bow(length, pp_dataframe, target='subjectivity'):
    '''From a dataframe of preprocessed tweets with annotated sentiment returns a padded
    sequence of one-hot vectors with corresponding classes
    :param: length, the maximum length of the sequence
    :param: pp_dataframe; dataframe with 'tokens', 'subjectivity', 'polarity' columns
    :type: pandas dataframe, 'tokens': list of strings, subjectivity and polarity are ints
    :param: classification; either 'subjectivity' or 'polarity'
    :type: string
    :return: sequence_vectors, m x l x n array where m is the number of example, l is the maximum length of the sequene and n is the dimension of the one-hot vector
    :rtype: numpy ndarray
    :return: Y, categorical response variables, 1 or 0, corresponding to each observation
    :rtype: numpy ndarray
    '''

    # set up token index map, X, Y as with bow
    raw_X, raw_Y = create_raw_X_Y(pp_dataframe, classification=target)
    X, Y = balance_data(raw_X, raw_Y)
    timap = token_index_map(X)
    num_tokens = max([timap[key] for key in timap]) + 1

    # for each observation set up a matrix of dimension l x n then populate the one-hot vectors
    # at the appropriate index adding each matrix to the array for the dataset
    # setup array for vectorised dataset
    sequence_bow = []
    # iterate through preprocessed dataset
    for observation in X:
        matrix = np.zeros([length, num_tokens])
        if len(observation) > length:
            observation = observation[:length]
        for i in range(len(observation)):
            token = observation[i]
            index = timap[token]
            matrix[i][index] = 1
        # add to dataset
        sequence_bow.append(matrix)
    # convert dataset to numpy arrays
    sequence_bow = np.array(sequence_bow)
    Y = np.array(Y)

    return sequence_bow, Y


def unknown_helper(model):
    '''Helper function to add the unknown token to a set of pretrained Gensim GloVe vectors.  The unknown token is the average of all the pretrained GloVe vectors as per Pennington advice
    :param: model, the pretrained GloVe vectors
    :type: Gensim KeyedVectors object
    :return: model including unknown token
    :rtype: Gensim KeyedVectors object
    '''
    embedding_size = len(model['bitcoin'])
    vec = np.mean([model[token] for token in list(model.vocab.keys())], axis=0)
    # add the vector to the saved GloVe vectors
    model.add(['<unk>'], [vec])
    return model


def embeddings(embedding_size, pp_dataframe, target='subjectivity'):
    '''Averages the GloVe vectors of specified dimension for each relevant observation replacing words not found in pre-trained vectors with the unknown word
    :param: embedding size (n): 50 or 200 (can be expanded by downloading and storing or even training new)
    :type: int
    :param: pp_dataframe, the preprocessed dataset
    :type: pandas dataframe
    :param: target either subjectivity or polarity
    :type: string
    :return: embeddings, the m x n matrix of embedddings, m observations, n dimensional embedding
    :rtype: numpy ndarray
    :return: Y, categorical response variables, 1 or 0, corresponding to each observation
    :rtype: numpy ndarray
    '''
    # extract and balance data from preprocessed text dataframe
    raw_X, raw_Y = create_raw_X_Y(pp_dataframe, classification=target)
    X, Y = balance_data(raw_X, raw_Y)
    # load pretrained and saved GloVe vectors
    modelname = f'{embeddings_directory}/twitter{str(embedding_size)}'
    model = KeyedVectors.load(modelname)
    # compute the unknown token
    model = unknown_helper(model)
    # iterate through preprocessed dataset averaging the word vectors for each set of tokens in the proprocessed text for each observation
    embeddings = []
    for tweet in X:
        vector = np.zeros(embedding_size)
        for token in tweet:
            # if the token exists in the pretrained vectors add it element wise to the total
            try:
                vector += model[token]
            # if the token is not in the pretrained vectors add the unknown word
            except KeyError:
                vector += model['<unk>']
        vector /= len(tweet)
        embeddings.append(vector)
    embeddings = np.array(embeddings)
    Y = np.array(Y)

    return embeddings, Y


def sequence_embedding(length, embedding_size, pp_dataframe, target='subjectivity'):
    '''Creates a sequence of GloVe word embeddings from the preprocessed tokens of each observation replacing with the unknown token and padding or truncating to desired length
    :param: length, the length of the sequence
    :type: int
    :param: embedding size, the dimension of the word embedding
    :type: int
    :param: pp_dataframe, the preprocessed dataset
    :type: pandas dataframe
    :param: target either subjectivity or polarity
    :type: string
    :return: embedding_sequence, m x l x n array (number of observations x length of sequence x dimension of embedding)
    :rtype: numpy ndarray
    :return: Y, the categorical response variable, 1 or 0, corresponding to each observation
    :rtype: numpy ndarray
    '''
    # extract and balance data from preprocessed text dataframe
    raw_X, raw_Y = create_raw_X_Y(pp_dataframe, classification=target)
    X, Y = balance_data(raw_X, raw_Y)
    # load pretrained and saved GloVe vectors
    modelname = f'{embeddings_directory}/twitter{str(embedding_size)}'
    model = KeyedVectors.load(modelname)
    # compute the unknown token
    model = unknown_helper(model)
    # iterate through the observations creating a sequence of vectors from each sequence of tokens
    embedding_sequences = []
    for tweet in X:
        embedding_sequence = np.zeros((length, embedding_size))
        if len(tweet) > length:
            tweet = tweet[:length]
        for i in range(len(tweet)):
            try:
                embedding_sequence[i] = model[tweet[i]]
            except KeyError:
                embedding_sequence[i] = model['<unk>']

        embedding_sequences.append(embedding_sequence)
    embedding_sequences = np.array(embedding_sequences)
    Y = np.array(Y)

    return embedding_sequences, Y


if __name__ == '__main__':
    # this section of code loads the pickled dataset of tweets, vectorises it and stores the vectorised dataset
    # it does not need to be re-run as the vectorised data is stored however can be
    for i in range(20):
        df = df = pd.read_pickle('tweet_data/pickled_datasets/subjectivity_data.pkl')
        longest_sample = max([len(tweet) for tweet in df['tokens']])
        average_sample = int(sum([len(tweet) for tweet in df['tokens']]) / len(df['tokens']))
        X_train, Y_train = embedding(50, df, target='subjectivity')
        np.save(f'{data_directory}/sub_emb50_X.npy', X_train)
        X = np.load(f'{data_directory}/sub_emb50_X.npy')
        np.save(f'{data_directory}/sub_emb50_Y.npy', Y_train)
        Y = np.load(f'{data_directory}/sub_emb50_Y.npy')
        print(np.allclose(X_train, X))
        print(np.allclose(Y_train, Y))
