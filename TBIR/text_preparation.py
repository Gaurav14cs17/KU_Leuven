__author__ = 'david_torrejon'

import os.path
from sys import stdout
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim
import logging
import time
from file_preparation import open_train_file_tfidf


class TbirText:

    def __init__(self, filepath = './data_6stdpt/DevData/scaleconcept16.teaser_dev_input_documents/docs/', is_dev=False, is_train=False, load_dict=True):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        if is_dev:
            filepath = './data_6stdpt/DevData/scaleconcept16.teaser_dev_input_documents/docs/'

        self.index = ''
        self.tfidf = ''
        self.is_train = is_train
        self.filepath = filepath
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.en_stop = get_stop_words('en')
        self.p_stemmer = PorterStemmer()
        self.doc_list = []
        #returns tokens in a list
        if self.is_train is False:
            for filename in os.listdir(self.filepath):
                stdout.write("\rloading file: %s" % filename)
                stdout.flush()
                doc_list.append(self.open_file(filename))
            self.dictionary = corpora.Dictionary(doc_list)
            self.dictionary.save('train_dict.dict')
        else:
            self.doc_list, self.file_n_list, self.query_id_list = open_train_file_tfidf(is_train = False, is_test = True)
            print ""
            if load_dict:
                self.dictionary=corpora.Dictionary.load('train_dict.dict')
            else:
                #print doc_list
                self.dictionary = corpora.Dictionary(self.doc_list)
                self.dictionary.save('train_dict.dict')

        print " "

        self.corpus = [self.dictionary.doc2bow(text) for text in self.doc_list]
        print 'corpus len', len(self.corpus)
        print 'dict len', len(self.dictionary)
        print self.dictionary


    def open_file(self, filename, test_file = False):
        """
            opens the file and returns the text in it
            @params

            returns:
                the text contained in the file
        """
        f_file = self.filepath+filename
        #print f_file
        try:
            if os.path.isdir(self.filepath):
                if os.path.isfile(f_file):
                    if test_file:
                        #print 'file found', f_file
                        None
                    f = open(f_file).read()
                    f = f.lower()
                    tokens = self.tokenizer.tokenize(f)
                    stopped_tokens = [i for i in tokens if not i in self.en_stop]
                    #stemmed_tokens = [self.p_stemmer.stem(i) for i in stopped_tokens]
                    return stopped_tokens
                else:
                    if test_file:
                        print 'file NOTTTT found', f_file
            else:
                if test_file:
                    print 'dir notert found', f_file

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)


    def test_one_file(self, filename):
            return self.open_file(filename, test_file = True)


    def train_model_lda(self, load_w=False, load_file = './lda_model_dev.3000'):
        """
            lda doesnt seem a good idea tbh
            @files to use
            lda_model_dev -> only on currently.
            lda_model_test
            @explanation
            get_document_topics(bow, minimum_probability=None)
            Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.

            Ignore topics with very low probability (below minimum_probability).

            get_topic_terms(topicid, topn=10)
        """

        if load_w is False:
            print 'building corpus...'
            print len(os.listdir(self.filepath))
            print 'generating model....'
            ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=3000, id2word = self.dictionary, passes=10, update_every=256)
            print 'model generated....'
            ldamodel.save(load_file)
            #print ldamodel.top_topics(corpus)
        else:
            ldamodel = gensim.models.ldamodel.LdaModel.load(load_file)
        #print ldamodel.print_topics
        print 'topics found: '
        #print len(ldamodel.show_topics(num_topics=3000, num_words=5))
        #print ldamodel.show_topics(num_topics=100, num_words=5)
        print 'workin on one file...'

        list_docs_token = []
        list_files = []
        trues = 0
        for filename in os.listdir(self.filepath):
                #print filename
            stdout.write("\rgenerating tokens for model lda: %s " % filename)
            stdout.flush()
            #doc_list.append(self.open_file(filename))
            one_file = self.test_one_file(filename) #removes stop_words
            #possibly wrong? returns the classification of the document and the tokens according to this classification

            #tokens_single_doc = [str(self.dictionary[tuple_t[0]]) for tuple_t in ldamodel[self.dictionary.doc2bow(one_file)]]
            tokens_single_doc = []
            for topic in ldamodel.get_document_topics(self.dictionary.doc2bow(one_file)):
                for term in ldamodel.get_topic_terms(topic[0]):
                    print self.dictionary[term[0]]
                    tokens_single_doc.append(self.dictionary[term[0]])
            #print 'file: '
            #print sorted([token for token in one_file])
            #print 'lda model: '

            #print sorted([token for token in tokens_single_doc])
            for token in tokens_single_doc:
                if token in one_file:
                    print 'true'
                    trues +=1
                    break
            #time.sleep(5)

            list_files.append(filename)
            list_docs_token.append(tokens_single_doc)
        print trues
        #print ldamodel[dictionary.doc2bow(one_file)]


        #print tokens_single_doc
        return list_docs_token, list_files
        #print ldamodel.top_topics(corpus, num_words=5)


    def train_model_tfidf(self, load_w=True, load_file = './tfidf_model_dev'):
        """
            try classification with BoW model?
            looks for similar texts of the provided returns the 100ths similarest documents with the tokens of these documents
        """
        if load_w is False:
            self.tfidf = gensim.models.tfidfmodel.TfidfModel(self.corpus)
            self.tfidf.save(load_file)
        else:
            self.tfidf = gensim.models.tfidfmodel.TfidfModel.load(load_file)


        self.index = similarities.SparseMatrixSimilarity(self.tfidf[self.corpus], num_features=(len(self.dictionary)))
        #stop here
        return len(self.doc_list)

    def classify_tfidf(self, batch_sz=1, iterations_done=0):
        list_docs_token = []
        list_files = []
        for i in range(batch_sz*iterations_done, batch_sz*iterations_done+batch_sz):
            #if i >= 5:
            #    break
            stdout.write("\rTEST generating tokens for model tfidf: %d " % i)
            stdout.flush()
            #print doc
            f = ' '.join(self.doc_list[i]).lower()
            tokens = self.tokenizer.tokenize(f)
            one_file = [i for i in tokens if not i in self.en_stop]
            bow_doc = self.dictionary.doc2bow(one_file)

            #print bow_doc
            #print index[tfidf[bow_doc]]
            #computes similarity to all documents in corpus
            sims = self.index[self.tfidf[bow_doc]]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            #print sims[:100]
            best_txts = []
            for i in range(0,1):
                #print '--------------------'
                text_to_return = []
                for token in self.corpus[sims[i][0]]:
                    #print self.dictionary[token[0]]
                    #print token
                    if token[1]>=1:
                        #print (self.dictionary[token[0]])
                        text_to_return.append(self.dictionary[token[0]])
            #print text_to_return
                best_txts.append(text_to_return)
            list_docs_token.append(best_txts)



        return list_docs_token, self.query_id_list, self.doc_list


    def train_model_lsi(self, load_w=True, load_file = './lsi_model_dev.200'):

        if load_w is False:
            lsi = gensim.models.lsimodel.LsiModel(corpus = self.corpus, num_topics=200, id2word=self.dictionary)
            lsi.save(load_file)
        else:
            lsi = gensim.models.lsimodel.LsiModel.load(load_file)

        list_docs_token = []
        list_files = []
        for filename in os.listdir(self.filepath):
                #print filename
            stdout.write("\rgenerating tokens for model lsi: %s " % filename)
            stdout.flush()
            one_file = self.test_one_file(filename) #bow of this file
            bow_doc = self.dictionary.doc2bow(one_file)

            one_file = self.test_one_file(filename) #bow of this file
            tokens_single_doc = [str(self.dictionary[tuple_t[0]]) for tuple_t in lsi[self.dictionary.doc2bow(one_file)]]
            list_files.append(filename)
            list_docs_token.append(tokens_single_doc)

        return list_docs_token, list_files


    def train_model_bow(self):

        list_docs_token = []
        list_files = []

        return list_docs_token, list_files

    def train_model_cca(self):
        """
        sci-kit learn look for it!
        """
