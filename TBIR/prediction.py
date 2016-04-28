__author__ = 'david_torrejon'
from PIL import Image
import cv2
import os.path
import operator
from sys import stdout
from stop_words import get_stop_words


class TbirPredict:

    def __init__(self, dictionary, is_dev = False, model = ''):
        assert model is not '', "model cant be empty"
        self.en_stop = get_stop_words('en')
        self.dictionary_words = dictionary
        self.is_dev = is_dev
        filename_output = 'output_classify_' + model + '.txt'
        print 'opening file for output', filename_output
        self.file_output = open(filename_output, 'w')
        self.filename_output = filename_output


    def predict_img(self, text,  filename_output, nb_imgs=100, show_img = False):
        """
            self.dictionary_words contains the relations key=imgid values=wordspairs
            Predicts an img according to the tokens on text, and according to the values of the dataset img-tokens given.
            @params
            text: tokenized processed text
            nb_images: number of similar images to retrieve, by default 1, the closest one.
            returns: image
            FORMAT OUTPUT query_id 0 img_id 0 score 0
        """

        if self.is_dev:
            imgs_directory = './data_6stdpt/DevData/scaleconcept16.teaser_dev_images/'
            tokens = text
        else:
            imgs_directory = './data_6stdpt/Features/Visual/scaleconcepts16_images/'
            if type(text) is not list:
                tokens = text.split(" ")
            else:
                tokens=[]
                try:
                    for t in text:
                        #print t
                        if t not in self.en_stop:
                            #print 'in'
                            tokens.append(str(t).encode('utf-8'))
                except:
                    for t in text:
                        if t not in self.en_stop:
                            #print 'in'
                            tokens.append(t)

        #print "\npredicting img for", tokens
        stdout.write("\rpredicting img for text: %s" % filename_output)
        stdout.flush()
        print tokens
        #image_predicted = Image.open()
        #image_predicted.show()
        #raise SystemExit(0)
        dict_img_retrieve={}
        for key, value in self.dictionary_words.items():
            """
                this part is really slow
                key is image id
                value is img word-pairs

                tokens - word token of the test line.
            """
            curr_val = 0
            curr_found = 0
            for pair in value:
                #print pair
                #print pair[0], tokens
                if pair[0] in tokens:
                    curr_found+=1
                    #print pair[0]
                    # curr_val += int(pair[1])/len(tokens) #weight on number of tokens wrong
                    """
                        The less tokens the better? if a word has a value of 100, but that image has 100 pairs
                        100pairs/100=1
                        but if the image has 50 pairs
                        50/100=0.5
                        meaning that the higher the value of the more number of tokens the better
                        though this provides the problem that if the img value is really low, provides a big value and can make the result go wrong.
                        if the value is worth 10 and there are 100 pairs
                        100/10=10, giving it a higher chance of success, maybe its not what we want.
                    """
                    #print curr_val
                    curr_val += int(pair[1])
                    #print pair
            #print curr_val
            if curr_val != 0:
                curr_val = (curr_val/float(100000))*curr_found/float(curr_val)
                if len(dict_img_retrieve) < nb_imgs:
                    dict_img_retrieve[key] = curr_val
                else:
                     #print dict_img_retrieve
                     #print curr_val, dict_img_retrieve[min(dict_img_retrieve, key=dict_img_retrieve.get)]
                     if curr_val > dict_img_retrieve[min(dict_img_retrieve, key=dict_img_retrieve.get)]:
                         del dict_img_retrieve[min(dict_img_retrieve, key=dict_img_retrieve.get)]
                         dict_img_retrieve[key] = curr_val

        sorted_dict = sorted(dict_img_retrieve.items(), key=operator.itemgetter(1), reverse=True)

        #print sorted_dict
        for key in sorted_dict:
            self.file_output = open(self.filename_output, 'a')
            self.file_output.write(filename_output) #query id
            self.file_output.write(" ")
            if self.is_dev:
                img = imgs_directory  + key[0] + '.jpg'
            else:
                img = imgs_directory + key[0][0:2].lower() + '/' + key[0] + '.jpg'
            self.file_output.write(key[0]) #img id
            self.file_output.write(" ")
            self.file_output.write(str(key[1])) #puntuation id
            self.file_output.write("\n")
            print filename_output, key[0], key[1]
            #print self.dictionary_words[key[0]]
            #print " "
            if os.path.isdir(imgs_directory):
                #print 'dir found'
                if os.path.isfile(img):
                    #print 'img found'
                    if show_img:
                        image_predicted = cv2.imread(img)
                        cv2.imshow(key[0], image_predicted)
                        cv2.waitKey(0)
                else:
                    stdout.write("\rimage not found: %s" % img)
                    stdout.flush()
            #stdout.write("\n")
        self.file_output.close()
        cv2.destroyAllWindows()
        #self.file_output.write("\n")
