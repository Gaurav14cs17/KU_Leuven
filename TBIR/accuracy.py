__author__ = 'david_torrejon'


def accuracy_predictions(predicted = './output_classify_tfidf.txt',
                        expected = './data_6stdpt/DevData/scaleconcept16.teaser_dev_id.v20160212/scaleconcept16.teaser_dev.ImgToTextID.txt'):

    """
    format of the files
        predicted: doc - img * n...
        expected: img - doc

    for now only works with 1 img as in predicted, then instead of 1 element, will add a list of imgs predicted
    and will search it with an in the list
    """

    dict_predicted = {}
    dict_exptected = {}
    with open(predicted) as fd:  #filedescriptor C ftw!!!
        for line in fd:
            k_v = line.rstrip().split(" ")
            if len(k_v) > 1:
                dict_predicted[k_v[0]]=k_v[1]

    with open(expected) as fd:  #filedescriptor C ftw!!!
        for line in fd:
            k_v = line.rstrip().split(" ")
            if len(k_v) > 1:
                dict_exptected[k_v[1]]=k_v[0]

    correct = 0

    #assert len(dict_exptected) == len(dict_predicted),  "length of data diferent (exptected =/= predicted)"
    #print len(dict_predicted), len(dict_exptected)
    for k in dict_predicted:
        #print dict_exptected[k], dict_predicted[k]
        if dict_exptected[k] == dict_predicted[k]:
            correct += 1
            #print correct
    print correct/float(len(dict_predicted))
