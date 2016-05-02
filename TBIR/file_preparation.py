__author__ = 'david_torrejon'

import os.path
from sys import stdout
from stop_words import get_stop_words
import numpy as np
import re
import pandas

"""
file_preparation module, picks the file questions_words.txt and processes it
it can be passed through line of command or else by default will try to get it from
the current dir.
"""

def create_output_dataset(input_filename, output_filename, solution_filename):
    """
        dataset is something liek
        a is to b what c is to do in format a b c d
        : explanation
        ...
        a b c d
        a b c d
        ...
        : exlanation
        ...
        so we will look for all lines that start with : and remove it
        and then for all lines remove d, and store it somwhere?

        v0.2 create as lists as : are, so you can search on them faster when looking at analogies...
        instead of when finding : skiping, write the category, so we have all
        organized by cathegories, and we are able to load on smaller lists that are faster to search

        TODO... load and smaller dataset just to test it instead of loading everything
        and then picking few expamples...
    """
    print('creating output and solutions set...')
    output_file = open(output_filename, 'w')
    solution_file = open(solution_filename, 'w')
    number_category = 1
    categories = []
    with open(input_filename) as fd:  #filedescriptor C ftw!!!
        for line in fd:
            if line[0] is not ':':
                split_line = line.split(" ")
                output_line = split_line[0] + ' ' + split_line[1] + ' ' + split_line[2] + '\n'
                output_file.write(str(output_line))
                solution_file.write(str(split_line[3]))
            else:
                category = str(number_category) + line[2:]
                print category
                categories.append(category)
                output_file.write(category)
                solution_file.write(category)
                number_category += 1

    return categories

def dataset_preparation(filename=None): #add outputfilename for dataset
    directory_try = False
    try:
        if filename is None:
            input_filename = './questions-words.txt'
            print('trying to open questions-words.txt from directory...')
            if os.path.isfile(input_filename):
                print 'found', input_filename, 'in directory...'
        else:
            print 'You provided the following route to the file: ', filename, '\n'
            input_filename = filename
        output_filename = './prepared-q-w.txt'
        solution_filename = './solution-q-w.txt'
        if os.path.isfile(output_filename) is False:
                create_output_dataset(input_filename, output_filename, solution_filename)
        else:
            print('the output dataset was previously created...\n') #maybe add remove option?
            recreate = raw_input('do you want to create it again...(y/n):   ')
            if recreate is 'y':
                print input_filename
                create_output_dataset(input_filename, output_filename, solution_filename)
    except:
        print('someting went wrong...')
        print 'maybe', filename, 'does not exist... let me try if the file exists in the directory...\n'
        if directory_try is False:
            directory_try = True
            dataset_preparation()
    finally:
        return output_filename, solution_filename



def process_line(line):
    """
        @params
        line: list of all tokens contained in a line
        format: id_img nb_pairs(word, points) w1 p1 w2 p2 .... wn pn
        return: key, value for the dictionary
        key: id_img
        value: list of pairs w-p
        remove stop words?
    """
    en_stop = get_stop_words('en')
    #print en_stop
    key = line[0]
    nb_pairs = int(line[1])
    i = 0
    value = []
    while i<nb_pairs*2:
        #print line[2+i]
        #if line[2+i] not in en_stop:
        value.append([line[2+i], int(line[3+i])])
        i+=2

    #assert nb_pairs == len(value), "length of data diferent (nb_pairs =/= len(pairs))"
    return key, value

def process_test_line(line):
    en_stop = get_stop_words('en')
    """
       proceses to generate a doc from the line. The weights wont be used this time.
    """
    i = 0
    key = line[0]
    nb_pairs = int(line[1])
    value = []
    while i<nb_pairs*2:
        value.append(line[2+i])
        i+=2
    assert nb_pairs == len(value), "length of data diferent (nb_pairs =/= len(pairs))"
    #print value
    return key, value

def open_train_datatxt(filename='./data_6stdpt/Features/Textual/train_data.txt', is_dev = False, is_train = False, is_test = True):
    """
        I dont know... try to open this file and see what to do?
        return a dictionary where the key is the id from image? the value a list of the pairs word-value
    """
    if is_dev:
        filename ='./data_6stdpt/DevData/scaleconcept16.teaser_dev_data_textual.scofeat.v20160212.david'
    train_data = {}
    if os.path.isfile(filename):
        print 'found', filename, 'in directory...'
        with open(filename) as fd:  #filedescriptor C ftw!!!
            for i,line in enumerate(fd):
                #if is_train is False:
                k, v = process_line(line.split(" "))
                train_data[k] = v
                #else:
                #    k, v = process_test_line(line.split(" "))
                #    train_data[k] = v
                if i%2000==0:
                    stdout.write("\rdatapoints loaded %s" % i)
                    stdout.flush()
#                raise SystemExit(0)
            stdout.write("\n")

    return train_data

def open_train_file_tfidf(filename = './data_6stdpt/train_data.txt', is_train = True, is_test = True):
    docs_list = []
    file_list = []
    query_id_list = []
    query_list = []
    if is_train:
        if os.path.isfile(filename):
            print 'found', filename, 'in directory...'
            with open(filename) as fd:  #filedescriptor C ftw!!!
                for i,line in enumerate(fd):
                    doc = []
                    file_n, tokens = process_test_line(line.split(" "))
                    docs_list.append(tokens)
                    file_list.append(file_n)
                    if i%2000==0:
                        stdout.write("\rdatapoints loaded %s" % i)
                        stdout.flush()
        print " "
    if is_test:
        filename='./data_6stdpt/test_data_students.txt'
        if os.path.isfile(filename):
            print 'found', filename, 'in directory...'
            with open(filename) as fd:  #filedescriptor C ftw!!!
                for i,line in enumerate(fd):
                    doc = []
                    file_n, tokens = process_test_line(line.split(" "))
                    query_list.append(tokens)
                    query_id_list.append(file_n)
                    if i%2000==0:
                        stdout.write("\rdatapoints loaded %s" % i)
                        stdout.flush()


    return docs_list, file_list, query_list, query_id_list

def process_line_mymodel(line):
    """
        @params
        line: list of all tokens contained in a line
        format: id_img nb_pairs(word, points) w1 p1 w2 p2 .... wn pn
        return: key, value for the dictionary
        key: id_img
        value: list of pairs w-p
        remove stop words?
    """
    en_stop = get_stop_words('en')
    #print en_stop
    key = line[0]
    nb_pairs = int(line[1])
    i = 0
    value = []
    weights = {}
    while i<nb_pairs*2:
        #print line[2+i]
        #if line[2+i] not in en_stop:
        value.append(re.sub(r'[^\x00-\x7f]',r'',line[2+i]))
        weights[re.sub(r'[^\x00-\x7f]',r'',line[2+i])]=int(line[3+i])
        i+=2

    #assert nb_pairs == len(value), "length of data diferent (nb_pairs =/= len(pairs))"
    return key, value, weights


def open_train_file_mymodel(dictionary_words_id, filename = './data_6stdpt/train_data.txt'):
    """
        returns matrix with nb_samples x len_dictionary. Each position has the WEIGHT of the word
        dictionary[id]=word
        also returns the map of id_img row on the matrix
    """
    matrix_weights = []
    idimg_to_row = {}
    if os.path.isfile(filename):
        print 'found', filename, 'in directory...'
        with open(filename) as fd:  #filedescriptor C ftw!!!
            for i,line in enumerate(fd):
                stdout.write("\rmatrix weights loaded %s" % i)
                stdout.flush()
                row = np.zeros(len(dictionary_words_id))
                file_n, tokens, weights = process_line_mymodel(line.split(" "))
                idimg_to_row[i]=file_n
                ids_to_rows = dictionary_words_id.doc2bow(tokens)
                word_ids = [dictionary_words_id[id_[0]] for id_ in ids_to_rows]
                #print weights
                #print weights['parker']
                for id_t in ids_to_rows:
                    row[id_t[0]] = weights[re.sub(r'[^\x00-\x7f]',r'',dictionary_words_id[id_t[0]])]
                #print row
                matrix_weights.append(row)

    return np.asarray(matrix_weights, dtype=float), idimg_to_row

def generate_dataset_inference(filetrain='./data_4stdpt/target_collection', filetest='./data_4stdpt/queries_val', outputfile='./premise_hypo.txt'):
    train_sentences = []
    test_sentences = []
    premises = pandas.read_table(filetrain)
    hypothesis = pandas.read_table(filetest)

    id_pre = premises['sent_id'].tolist()
    id_hypo = hypothesis['sent_id'].tolist()

    premises_list = premises['txt_caption'].tolist()
    hypothesis_list = hypothesis['txt_caption'].tolist()
    img_pre = premises['img_id'].tolist()
    img_hyp = hypothesis['img_id'].tolist()

    output = open(outputfile, 'w')
    lines_written = 0
    same = 0
    for hypothesi, id_hyp_img, id_hypo_query in zip(hypothesis_list, img_hyp, id_hypo):
        for premise, id_pre_img, id_pre_query in zip(premises_list, img_pre, id_pre):
            if id_hyp_img == id_pre_img:
                same+=1
                line_to_write = str(id_hypo_query) + "#&#" + hypothesi + "#&#" + premise + "#&#" + str(id_pre_img) +"#&#" + str(id_pre_query) +"\n"
            else:
                line_to_write = str(id_hypo_query) + "#&#" + hypothesi + "#&#" + premise + "#&#" + "wrong" + "#&#" + str(id_pre_query) +"\n"
            #print line_to_write
            output.write(line_to_write)
            lines_written+=1
        #raise SystemExit(0)
    print lines_written, same, len(set(img_hyp)), len(set(img_pre))


def test_dataset_inference(outputfile='./premise_hypo.txt'):
    entailments = 0
    with open(outputfile) as fd:  #filedescriptor C ftw!!!
        for line in fd:
            t_data = line.rstrip().split("#&#")
            print t_data
            raise SystemExit(0)
            if 'entailment' in t_data:
                entailments+=1
    print entailments



def generate_dataset(filetrain='./data_4stdpt/target_collection', filetest='./data_4stdpt/queries_val', outputfile='./premise_hypo.txt'):
    train_sentences = []
    test_sentences = []

    premises = pandas.read_table(filetrain)
    hypothesis = pandas.read_table(filetest)
    id_pre = premises['sent_id'].tolist()
    id_hypo = hypothesis['sent_id'].tolist()
    premises_list = premises['txt_caption'].tolist()
    hypothesis_list = hypothesis['txt_caption'].tolist()
    img_pre = premises['img_id'].tolist()
    img_hyp = hypothesis['img_id'].tolist()

    for hypothesi, id_hyp_img, id_hypo_query in zip(hypothesis_list, img_hyp, id_hypo):
        test_sentences.append([id_hypo_query, id_hyp_img, hypothesi])
    for premise, id_pre_img, id_pre_query in zip(premises_list, img_pre, id_pre):
        train_sentences.append([id_pre_query, id_pre_img, premise])

    return train_sentences, test_sentences
