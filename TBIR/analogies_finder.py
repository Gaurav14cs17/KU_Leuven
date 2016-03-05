__author__ = 'david_torrejon'

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine(a, b):
    """
     @params
     a: numpy array
     b: numpy array
     returns the cosine similarity between these parameters
    """
    return cosine_similarity(a, b)


def transformation_to_positive(a):
    a_transformed = np.zeros(len(a))
    for i in range(len(a)):
        a_transformed[i] = (a[i]+1)/2

    return a_transformed


def get_cosine_similarity(a, b, c, d):
    """
     @params
     a: numpy array
     b: numpy array
     c: numpy array
     d: numpy array
     returns the cosine similarity between these parameters

     need to transform all elements to (x+1)/2 to make them positive for the
     multiplication method
    """
    epsilon = 0.001

    a_transf = transformation_to_positive(a)
    b_transf = transformation_to_positive(b)
    c_transf = transformation_to_positive(c)
    d_transf = transformation_to_positive(d)

    a_transf = np.reshape(a_transf, (1,-1))
    b_transf = np.reshape(b_transf, (1,-1))
    c_transf = np.reshape(c_transf, (1,-1))
    d_transf = np.reshape(d_transf, (1,-1))

    dc = cosine(d_transf, c_transf)
    db = cosine(d_transf, b_transf)
    da = cosine(d_transf, a_transf)

    return (dc*db)/(da+epsilon)



def build_glove_dictionary():
    """
        builds a dictionary based on the glove model.
        http://nlp.stanford.edu/projects/glove/
        dictionary will have the form of key = token, value = numpy array with the pretrained values
    """
    print ('building glove dictionary...')
    glove_file = './glove.6B.50d.txt'
    glove_dict = {}
    with open(glove_file) as fd_glove:
        for i, input in enumerate(fd_glove):
            if i%5000 == 0:
                print i, 'entries on the dictionary'
            input_split = input.split(" ")
            #print input_split
            key = input_split[0] #get key
            del input_split[0]  # remove key
            values = []
            for value in input_split:
                values.append(float(value))
            np_values = np.asarray(values)
            glove_dict[key] = np_values

    print 'dictionary build with length', len(glove_dict)

    return glove_dict

def build_outputs_list(filename):
    outputs = []
    with open(filename) as fd_outputs:
        for output in fd_outputs:
            outputs.append(output)
    return outputs


def find_analogies(input_file, ouptut_file):

    """
        build analogy finder?
        input file contains a b c
        and output file d
        d = c - a + b

        I will read from file...but should be faster to load all to memory and work from there...
        maybe I change it later on
        gotta load glove to ram, for uses sake
        should add a counter for the a b c found in the glove model!!
    """

    glove_model = build_glove_dictionary() #returns obviously the glove dictionary
    possible_outputs = build_outputs_list(ouptut_file) #d'
    analogies_computed = 0

    with open(input_file) as fd_inputs:  #
        for input in fd_inputs:
            split_input = input.split(" ")
            a, b, c = split_input[0], split_input[1], split_input[2].rstrip('\n')
            d = np.zeros(len(a))
            max_d = 0
            output = None
            print 'trying...', a, b, c
            if (a in glove_model) and (b in glove_model) and (c in glove_model):
                for d_poo in possible_outputs:
                    d_po = d_poo.rstrip('\n')
                    if d_po in glove_model:
                        #print d_po, 'found...'
                        analogies_computed += 1
                        d_current = get_cosine_similarity(glove_model[a], glove_model[b], glove_model[c], glove_model[d_po])
                        if d_current > max_d:
                            max_d = d_current
                            output = d_po
                    else:
                        #make this a lil bit more efficient....
                        possible_outputs.remove(d_poo)
            print 'output found: ', output

    print analogies_computed
    #compute cosine similarity and argmax...
