__author__ = 'david_torrejon'

from file_preparation import generate_dataset_inference, test_dataset_inference, generate_dataset
from accuracy_4pt import check_accuracy, compare_toqueries
from text_preparation import TbirText
from language_models import generate_ngrams, generate_id_sentence, train_ngram, generate_train_ngram_score, generate_matrix
from gensim import utils
from analogies_finder import cosine

import numpy as np

#generate_dataset_inference()
#test_dataset_inference()
#check_accuracy()
#compare_toqueries()
def find_nearest(array,value=0):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


trainset, testset = generate_dataset()

matrix_train, token_dict = generate_ngrams(trainset)
print matrix_train.shape

tokenized_ts=[utils.simple_preprocess(s[2], max_len=20) for s in testset]
matrix_test = generate_matrix(tokenized_ts, token_dict, maxlen=len(token_dict), is_train=False)
print matrix_test.shape
result = cosine(matrix_test, matrix_train)
print result.shape
precision = 0
output = open("output_cosine_similarity.txt", 'w')
for i, array in enumerate(result):
    idx = np.argmax(array)
    output_s = trainset[idx][1] + " " +testset[i][1] + "\n"
    output.write(output_s)
    if trainset[idx][1] == testset[i][1]:
        precision+=1
print "precision", precision

"""
dict_counts = train_ngram(trainset)
tokenized_ts=[utils.simple_preprocess(s[2], max_len=20) for s in trainset]
print "train"
train_array = generate_train_ngram_score(dict_counts, tokenized_ts)
print "test"
tokenized_ts=[utils.simple_preprocess(s[2], max_len=20) for s in testset]
test_array = generate_train_ngram_score(dict_counts, tokenized_ts, bool_train=False)

scores = train_array - test_array
print scores.shape
output = open("output_unigrams.txt", 'w')
precision = 0
for i, array in enumerate(scores):
    value, idx = find_nearest(array, 0)
    output_s = trainset[idx][1] + " " +testset[i][1] + "\n"
    output.write(output_s)
    if trainset[idx][1] == testset[i][1]:
        precision+=1

print "precision", precision
"""
