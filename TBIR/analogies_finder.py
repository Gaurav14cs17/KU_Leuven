__author__ = 'david_torrejon'

import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from random import seed, uniform

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


def get_cosine_similarity(a, b, c, d, similarity_type="multiplication"):
    """
     @params
     a: numpy array
     b: numpy array
     c: numpy array
     d: numpy array
     returns the cosine similarity between these parameters
     return multiplication, addition, direction

     need to transform all elements to (x+1)/2 to make them positive for the
     multiplication method

     v0.2 add diferent methos like direction or addition
     for fastest comparations returns the 3 models, instead of goin through the file 3 times....

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
    """
    addition
    cos(d,c-a+b)
    """
    a_t = np.reshape(a, (1,-1))
    b_t = np.reshape(b, (1,-1))
    c_t = np.reshape(c, (1,-1))
    d_t = np.reshape(d, (1,-1))


    sum_result = c_t - a_t + b_t
    """
    direction
    cos(a-b, c-d)
    """

    return (dc*db)/(da+epsilon), cosine(d_t, sum_result), cosine(a_t-b_t, c_t-d_t)


def build_glove_dictionary():
    """
        builds a dictionary based on the glove model.
        http://nlp.stanford.edu/projects/glove/
        dictionary will have the form of key = token, value = numpy array with the pretrained values

        REALLY IMPORTANT the glove dataset. with the big one finds nearly everything....
        smallest one...quite baaaaaad...
    """
    print ('building glove dictionary...')
    glove_file = './glove.840B.300d.txt'
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
    """
        v0.1 returned 1 big list
        v0.2 returns dict of lists depending on category
    """
    outputs = {}
    expected = []
    category = ""
    first_category = True #controls FIRST CATEGORY ONLY
    category_output= []

    with open(filename) as fd_outputs:
        for output in fd_outputs:
            if output[0].isdigit():
                #print output
                #save in the dictionary
                if first_category is False:
                    outputs[category] = category_output
                else:
                    first_category = False
                #rebuild again for next category
                category = output
                category_output = []
            else:
                category_output.append(output)
                expected.append(output)

    outputs[category] = category_output

    return outputs, expected


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
    seed = 1337 #l33t

    percentage_test = 0.1 #make this optional or maybe a parameter for the person to decide?

    glove_model = build_glove_dictionary() #returns obviously the glove dictionary
    possible_outputs_dict, expected_output = build_outputs_list(ouptut_file) #d'
    #print possible_outputs_dict.keys()
    analogies_computed = 0

    correct_mult = 0
    correct_add = 0
    correct_dir = 0

    line_counter = 0 #this refers to the current output that we are looking for

    with open(input_file) as fd_inputs:  #
        for input in fd_inputs:
            #print input, expected_output[line_counter]
            test_accepted = False
            if input[0].isdigit() is False:

                num = uniform(1.0, 0.0)
                category_bool = False

                if num<percentage_test:
                    test_accepted = True
                    split_input = input.split(" ")
                    a, b, c = split_input[0], split_input[1], split_input[2].rstrip('\n')
                    d = np.zeros(len(a))
                    max_d_mult = 0
                    max_d_add = 0
                    max_d_dir  = 0
                    output_mult = None
                    output_add = None
                    output_dir = None
                    print 'trying...', a, b, c
                    start = time.time()
                    if (a in glove_model) and (b in glove_model) and (c in glove_model):
                        for d_poo in possible_outputs:
                            d_po = d_poo.rstrip('\n')
                            if d_po in glove_model:
                                #print d_po, 'found...'
                                analogies_computed += 1
                                # pass another value depending on which type of similarity is applied
                                d_current_mult, d_current_add, d_current_dir  = get_cosine_similarity(glove_model[a], glove_model[b], glove_model[c], glove_model[d_po])
                                #so pretty
                                if d_current_mult > max_d_mult:
                                    max_d_mult = d_current_mult
                                    output_mult = d_po
                                if d_current_add > max_d_add:
                                    max_d_add = d_current_add
                                    output_add = d_po
                                if d_current_dir > max_d_dir:
                                    max_d_dir = d_current_dir
                                    output_dir = d_po
                            else:
                                #make this a lil bit more efficient....
                                possible_outputs.remove(d_poo)
                line_counter += 1
            else:
                category_bool = True
                possible_outputs = possible_outputs_dict[input]

            if category_bool is False and test_accepted:
                print 'output(mult) found: ', output_mult, 'expected: ', expected_output[line_counter-1]
                print 'output(add) found: ', output_add, 'expected: ', expected_output[line_counter-1]
                print 'output(dir) found: ', output_dir, 'expected: ', expected_output[line_counter-1]
                if output_mult is expected_output[line_counter-1]:
                    correct_mult +=1
                if output_add is expected_output[line_counter-1]:
                    correct_add +=1
                if output_dir is expected_output[line_counter-1]:
                    correct_dir +=1



    print 'multi correct: ', correct_mult/analogies_computed
    print 'add correct: ', correct_add/analogies_computed
    print 'dir correct: ', correct_dir/analogies_computed
    #compute cosine similarity and argmax...
