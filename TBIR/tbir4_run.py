__author__ = 'david_torrejon'

from file_preparation import generate_dataset_inference, test_dataset_inference, generate_dataset
from accuracy_4pt import check_accuracy, compare_toqueries, compute_precision_recall_ap, compute_MAP, compute_AP
from text_preparation import TbirText
from language_models import generate_ngrams, generate_id_sentence, train_ngram, generate_train_ngram_score, generate_matrix, generate_sum_words, generate_tfidf_model, generate_lda_model
from gensim import utils
from analogies_finder import cosine, build_glove_dictionary
from matplotlib import pyplot as plt

import numpy as np

#generate_dataset_inference()
#test_dataset_inference()
#check_accuracy()
#compare_toqueries()

def find_nearest(array,value=0):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


trainset, testset = generate_dataset(test_submission=True)

matrix_train, token_dict = generate_ngrams(trainset)
print matrix_train.shape
"""
lda
"""
lda = False
if lda:
    result = generate_lda_model(trainset, testset)
    rec_list = []
    prec_list = []
    aps = []
    output_file_sub = open("submission_lda.txt", 'w')
    for i, array in enumerate(result):
        idx = [iid[0] for iid in array]
        for id_ in array:
            """
                query_id 0 sent_id 0 score 0
            """
            o_sub = str(testset[i][0]) + " 0 " + str(trainset[id_[0]][0]) + " 0 " + str(id_[1]) + " 0\n"
            output_file_sub.write(o_sub)
        #print idx
        #raise SystemExit(0)
        p, r, ap = compute_precision_recall_ap(idx, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)

    print compute_MAP(aps)
    #raise SystemExit(0)

    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()




"""
tfidf
"""
tfidf = True
if tfidf:
    result = generate_tfidf_model(trainset, testset)
    rec_list = []
    prec_list = []
    aps = []
    output_file_sub = open("submission_tf_idf.txt", 'w')
    for i, array in enumerate(result):
        idx = [iid[0] for iid in array]
        for id_ in array:
            """
                query_id 0 sent_id 0 score 0
            """
            o_sub = str(testset[i][0]) + " 0 " + str(trainset[id_[0]][0]) + " 0 " + str(id_[1]) + " 0\n"
            output_file_sub.write(o_sub)
        #print idx
        #raise SystemExit(0)
        p, r, ap = compute_precision_recall_ap(idx, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)

    print compute_MAP(aps)
    #raise SystemExit(0)


    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()


"""
generate word-sums
"""
word_sums = False
if word_sums:
    print "word-sums-mult"
    glove_model = build_glove_dictionary() #returns obviously the glove dictionary
    s_glove_train, s_glove_test, m_glove_train, m_glove_test = generate_sum_words(trainset, testset, glove_model)

    print s_glove_train.shape, s_glove_test.shape
    result = cosine(s_glove_test, s_glove_train)
    print result.shape
    rec_list = []
    prec_list = []
    aps = []
    output_file_sub = open("submission_word_sums.txt", 'w')
    for i, array in enumerate(result):

        # pick relevant 1000
        # query_id 0 img_id 0 score 0
        idx = array.argsort()[-1000:][::-1]
        for id_ in idx:
            """
                query_id 0 sent_id 0 score 0
            """
            o_sub = str(testset[i][0]) + " 0 " + str(trainset[id_][0]) + " 0 " + str(array[id_]) + " 0\n"
            output_file_sub.write(o_sub)

        p, r, ap = compute_precision_recall_ap(idx, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)
    print "sums"
    print compute_MAP(aps)

    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()


    result = cosine(m_glove_test, m_glove_train)
    print result.shape
    rec_list = []
    prec_list = []
    aps = []
    output_file_sub = open("submission_word_mult.txt", 'w')
    for i, array in enumerate(result):

        # pick relevant 1000
        # query_id 0 img_id 0 score 0
        idx = array.argsort()[-1000:][::-1]
        for id_ in idx:
            """
                query_id 0 sent_id 0 score 0
            """
            o_sub = str(testset[i][0]) + " 0 " + str(trainset[id_][0]) + " 0 " + str(array[id_]) + " 0\n"
            output_file_sub.write(o_sub)

        p, r, ap = compute_precision_recall_ap(idx, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)

    print "mult"
    print compute_MAP(aps)

    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()




#LSTMs

lstm = False
if lstm:
    result = check_accuracy()
    print len(result)
    #print result
    rec_list = []
    prec_list = []
    aps = []
    for i, r in enumerate(result):
        print len(r)
        array = [int(x[1]) for x in r]
        #print array
        p, r, ap = compute_precision_recall_ap(array, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)

    print compute_MAP(aps)

    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()



"""
Sentence vectorization
"""
tf_sent = False
if tf_sent:
    tokenized_ts=[utils.simple_preprocess(s[2], max_len=20) for s in testset]
    matrix_test = generate_matrix(tokenized_ts, token_dict, maxlen=len(token_dict), is_train=False)
    print matrix_test.shape
    result = cosine(matrix_test, matrix_train)
    print result.shape
    output = open("output_cosine_similarity.txt", 'w')
    rec_list = []
    prec_list = []
    aps = []
    output_file_sub = open("submission.txt", 'w')
    for i, array in enumerate(result):

        # pick relevant 1000
        # query_id 0 img_id 0 score 0
        idx = array.argsort()[-1000:][::-1]
        for id_ in idx:
            """
                query_id 0 sent_id 0 score 0
            """
            o_sub = str(testset[i][0]) + " 0 " + str(trainset[id_][0]) + " 0 " + str(array[id_]) + " 0\n"
            output_file_sub.write(o_sub)

        p, r, ap = compute_precision_recall_ap(idx, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)

    print compute_MAP(aps)


    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()

unigram=False
if unigram:
    dict_counts = train_ngram(trainset)
    tokenized_ts=[utils.simple_preprocess(s[2], max_len=20) for s in trainset]
    print "train"
    train_array = generate_train_ngram_score(dict_counts, tokenized_ts)
    print "test"
    tokenized_ts=[utils.simple_preprocess(s[2], max_len=20) for s in testset]
    test_array = generate_train_ngram_score(dict_counts, tokenized_ts, bool_train=False)
    result = train_array - test_array
    result = np.abs(result)
    print result.shape
    output = open("output_unigram.txt", 'w')
    rec_list = []
    prec_list = []
    aps = []

    for i, array in enumerate(result):
        # pick relevant 1000
        # query_id 0 img_id 0 score 0
        idx = array.argsort()[-1000:][::-1]
        for id_ in idx:
            """
                query_id 0 sent_id 0 score 0
            """
            o_sub = str(testset[i][0]) + " 0 " + str(trainset[id_][0]) + " 0 " + str(array[id_]) + " 0\n"
            output.write(o_sub)

        p, r, ap = compute_precision_recall_ap(idx, testset[i], trainset)
        rec_list.append(r)
        prec_list.append(p)
        aps.append(ap)
        #raise SystemExit(0)

    print compute_MAP(aps)


    plt.clf()
    plt.plot(rec_list[5], prec_list[5],'-r' , rec_list[438], prec_list[438], '-b', rec_list[676], prec_list[676], '-g', rec_list[783], prec_list[783],'-m', rec_list[924], prec_list[924],'-y')
    label = "queries 5 r, 438 b, 676 g, 783 m, 924 y"
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(label)
    plt.axis([0, 1.0, 0, 1.0])
    #plt.legend(loc="lower left")
    plt.show()
