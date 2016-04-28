__author__ = 'david_torrejon'

from file_preparation import open_train_datatxt, open_train_file_mymodel
from prediction import TbirPredict
from text_preparation import TbirText
#from redo_dataset import fredo_dataset
from accuracy import accuracy_predictions, compare_models
import operator
from image_similarity import Img_Tbir

def asint(s):
    try: return int(s), ''
    except ValueError: return sys.maxint, s

bool_compare_only = False
train = True
test  = False
is_dev = False
which_model = 'word2vec'
#fredo_dataset()

img_model = Img_Tbir()



model = TbirText(is_dev=is_dev, is_train=True)
#open_train_file_mymodel(dictionary_words_id=model.dictionary)
#print 'working on tfidf model'
#nb_docs = model.train_model_tfidf(load_w=False)
nb_docs = model.train_model_word2vec(load_w=False)
print 'generating dictionary imgid - wordpairs \n'
dict_text = open_train_datatxt(is_dev=is_dev, is_train = False)
pred = TbirPredict(dict_text, is_dev=is_dev, model = which_model)

print 'training....?'
#print list_files
batch_sz=50
iterations = nb_docs/batch_sz
print iterations
for i in range(iterations):
    #docs_tokens, list_files, queries = model.classify_tfidf(batch_sz=batch_sz, iterations_done=i)
    docs_tokens, list_files, queries = model.classify_model_word2vec(batch_sz=batch_sz, iterations_done=i)
    for file_, filename, query in zip(docs_tokens, list_files, queries):
        print 'query_id: ', filename, 'query: ', ' '.join(query)
        img = pred.predict_img(file_, filename_output=filename, nb_imgs=100)
        img_score = img_model.image_similarity(img1=img)
        pred.write_file_predictions(filename, img_score)

"""
if bool_compare_only is False:

    model = TbirText(is_dev=is_dev, is_train=True)


    if 'lda' in which_model:
        print 'working on lda model'
        docs_tokens, list_files = model.train_model_lda(load_w=True)
    if 'tfidf' in which_model:

    if 'lsi' in which_model:
        print 'working on lsi model'
        docs_tokens, list_files = model.train_model_lsi(load_w=False)
    #print docs_tokens



    #print len(docs_tokens)
    #print docs_tokens[0]
    if train:
        print 'training....?'
        #print list_files
        batch_sz=50
        iterations = nb_docs/batch_sz
        print iterations
        for i in range(iterations):
            docs_tokens, list_files, queries = model.classify_tfidf(batch_sz=batch_sz, iterations_done=i)
            for files, filename, query in zip(docs_tokens, list_files, queries):
                print query
                for tokens in files:
                    pred.predict_img(tokens, filename_output=filename, nb_imgs=100)

        #print "\ncomputing accuracy train for model", which_model
        #file_with_predictions = './output_classify_' + which_model + '.txt'
        #accuracy_predictions(predicted = file_with_predictions)

#compare_models(model_1 = 'lda', model_2 = 'lsi')

dict_text_test = open_train_datatxt(filename='./data_6stdpt/test_data_students.txt', is_train = True) #testfile dict[0] = [pairs]

# do the predictions!
if test:
    for id_query in sorted(dict_text_test, key=asint):
        #print id_query, dict_text_test[id_query]
        pred.predict_img(dict_text_test[id_query], filename_output=id_query, nb_imgs=100)
    print "\ncomputing accuracy test for model", which_model
        #raise SystemExit(0)
"""
