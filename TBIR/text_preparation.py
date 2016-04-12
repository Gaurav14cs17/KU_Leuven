__author__ = 'david_torrejon'

import os.path
from sys import stdout
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim
import logging



class TbirText:

    def __init__(self, filepath = './data_6stdpt/DevData/scaleconcept16.teaser_dev_input_documents/docs/'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.filepath = filepath
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.en_stop = get_stop_words('en')
        self.p_stemmer = PorterStemmer()
        doc_list = []
        #returns tokens in a list
        for filename in os.listdir(self.filepath):
            stdout.write("\rloading file: %s" % filename)
            stdout.flush()
            doc_list.append(self.open_file(filename))

        print " "
        self.dictionary = corpora.Dictionary(doc_list)
        self.corpus = [self.dictionary.doc2bow(text) for text in doc_list]
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


    def train_model_lda(self, load_w=False, load_file = './lda_model_dev.150'):
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
            ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=150, id2word = self.dictionary, passes=50)
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
        for filename in os.listdir(self.filepath):
                #print filename
            stdout.write("\rgenerating tokens for model lda: %s " % filename)
            stdout.flush()
            #doc_list.append(self.open_file(filename))
            one_file = self.test_one_file(filename) #bow of this file
            tokens_single_doc = [str(self.dictionary[tuple_t[0]]) for tuple_t in ldamodel[self.dictionary.doc2bow(one_file)]]
            list_files.append(filename)
            list_docs_token.append(tokens_single_doc)
        #print ldamodel[dictionary.doc2bow(one_file)]


        #print tokens_single_doc
        return list_docs_token, list_files
        #print ldamodel.top_topics(corpus, num_words=5)


    def train_model_tfidf(self, load_w=False, load_file = './tfidf_model_dev'):
        """
            try classification with BoW model?
        """
        if load_w is False:
            tfidf = gensim.models.tfidfmodel.TfidfModel(self.corpus)
            tfidf.save(load_file)
        else:
            tfidf = gensim.models.tfidfmodel.TfidfModel.load(load_file)


        index = similarities.SparseMatrixSimilarity(tfidf[self.corpus], num_features=(len(self.dictionary)))

        list_docs_token = []
        list_files = []
        for filename in os.listdir(self.filepath):
                #print filename
            stdout.write("\rgenerating tokens for model tfidf: %s " % filename)
            stdout.flush()
            one_file = self.test_one_file(filename) #bow of this file
            bow_doc = self.dictionary.doc2bow(one_file)

            sims = index[tfidf[bow_doc]]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])

            text_to_return = []
            for token in self.corpus[sims[0][0]]:
                #print self.dictionary[token[0]]
                if token[1]>1:
                    text_to_return.append(str(self.dictionary[token[0]]))
            list_docs_token.append(text_to_return)
            list_files.append(filename)

        return list_docs_token, list_files
