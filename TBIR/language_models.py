__author__ = 'david_torrejon'

from gensim import utils
import numpy as np
from collections import defaultdict

def generate_matrix(sentences, token_dict, maxlen=20, is_train=True):
    """
    """
    matrix=[]
    for sentence in sentences:
        sentence_id = generate_id_sentence(sentence, token_dict, maxlen)
        matrix.append(sentence_id)
    return np.asarray(matrix)

def generate_id_sentence(sentence, token_dict, maxlen=20):
    sentence_id = np.zeros(maxlen)
    for token in sentence:
        try:
            sentence_id[token_dict.keys()[token_dict.values().index(token.lower())]]+=1
        except:
            print "unknown token"
    return sentence_id

def generate_dictionary(docs):
    """
     a document is a collection of tokens.
     returns set of all tokens. key = id value = word
    """
    token_dict = {}
    words_set = set()
    for doc in docs:
        for token in doc:
            if token.lower() not in words_set:
                token_dict[len(words_set)]=token.lower()
                words_set.add(token.lower())
    return token_dict


def generate_ngrams(train_set, n=1):
    """
    split sentence in tokens, compare how many tokens are one in another?
    generates sentences of max 15 tokens
    """
    sentences_tokens = [utils.simple_preprocess(s[2], max_len=20) for s in train_set]
    token_dict = generate_dictionary(sentences_tokens)
    matrix_train = generate_matrix(sentences_tokens, token_dict, maxlen=len(token_dict), is_train=True)
    return matrix_train, token_dict

def train_ngram(train_set, n=1):
    dict_counts = defaultdict(int)
    total_counts = 0
    prob_1 = 0 #smoothing
    sentences_tokens = [utils.simple_preprocess(s[2], max_len=20) for s in train_set]
    for sentence in sentences_tokens:
        sentence.append("</s>")

    for sentence in sentences_tokens:
        for token in sentence:
            #print token
            dict_counts[token]+=1
            total_counts+=1

    for key, value in dict_counts.items():
        if dict_counts[key] == 1:
            prob_1 = dict_counts[key]
        dict_counts[key] = dict_counts[key]/float(total_counts)

    return dict_counts


def generate_train_ngram_score(dict_counts, train_sentences, bool_train=True):
    #generate log probs
    array_probabilities = []
    for sentence in train_sentences:
        value = 0
        for token in sentence:
            value -= np.log(dict_counts[token])
        array_probabilities.append(value)
    if bool_train:
        return np.asarray(array_probabilities).reshape(1,-1)
    else:
        return np.asarray(array_probabilities).reshape(-1,1)
