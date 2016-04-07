__author__ = 'david_torrejon'

import os.path
from sys import stdout
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim



class TbirText:

    def __init__(self, filepath = './data_6stdpt/DevData/scaleconcept16.teaser_dev_input_documents/docs/'):
        self.filepath = filepath
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.en_stop = get_stop_words('en')
        self.p_stemmer = PorterStemmer()

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
                        print 'file found', f_file
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



    def train_model_lda(self, load_w=True, load_file = './lda_model'):
        """
            get_document_topics(bow, minimum_probability=None)
            Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.

            Ignore topics with very low probability (below minimum_probability).

            get_topic_terms(topicid, topn=10)
        """
        doc_list = []

        for filename in os.listdir(self.filepath):
                #print filename
            stdout.write("\rloading file: %s" % filename)
            stdout.flush()
            doc_list.append(self.open_file(filename))

        print ""
        dictionary = corpora.Dictionary(doc_list)
        corpus = [dictionary.doc2bow(text) for text in doc_list]
        if load_w is False:
            print 'building corpus...'
            print len(os.listdir(self.filepath))
            print 'generating model....'
            ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=20)
            print 'model generated....'
            ldamodel.save('lda_model')
            print ldamodel.top_topics(corpus)
        else:
            ldamodel = gensim.models.ldamodel.LdaModel.load(load_file)
        #print ldamodel.print_topics
        print 'topics found: '
        print len(ldamodel.show_topics(num_topics=100, num_words=5))
        #print ldamodel.show_topics(num_topics=100, num_words=5)
        print 'workin on one file...'
        one_file = self.test_one_file('_0mcdtHvA8PsGtEI') #bow of this file
        #print ldamodel[dictionary.doc2bow(one_file)]

        tokens_single_doc = [str(dictionary[tuple_t[0]]) for tuple_t in ldamodel[dictionary.doc2bow(one_file)]]
        print tokens_single_doc
        return tokens_single_doc
        #print ldamodel.top_topics(corpus, num_words=5)
