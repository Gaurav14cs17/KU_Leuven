__author__ = 'david_torrejon'
from PIL import Image
import cv2
import os.path
import operator
from sys import stdout

class TbirPredict:

    def __init__(self, dictionary, is_dev = False):
        self.dictionary_words = dictionary
        self.is_dev = is_dev
        self.file_output = open('output_classify_lda.txt', 'w')
        self.file_output.write("input          output\n")
        self.file_output.write("---------------------\n")

    def predict_img(self, text,  filename_output, nb_imgs=1, show_img = False):
        """
            Predicts an img according to the tokens on text, and according to the values of the dataset img-tokens given.
            @params
            text: tokenized processed text
            nb_images: number of similar images to retrieve, by default 1, the closest one.
            returns: image
        """
        self.file_output.write(filename_output)
        self.file_output.write(" ")
        if self.is_dev:
            imgs_directory = './data_6stdpt/DevData/scaleconcept16.teaser_dev_images/'
            tokens = text
        else:
            imgs_directory = './data_6stdpt/Features/Visual/scaleconcepts16_images/'
            tokens = text.split(" ")

        print "\npredicting img for", tokens
        #image_predicted = Image.open()
        #image_predicted.show()
        #raise SystemExit(0)
        dict_img_retrieve={}
        for key, value in self.dictionary_words.items():
            curr_val = 0
            for pair in value:
                if pair[0] in tokens:
                    curr_val += int(pair[1])/len(tokens) #weight on number of tokens
                    #print pair
            #print curr_val
            if curr_val != 0:
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
            if self.is_dev:
                img = imgs_directory  + key[0] + '.jpg'
            else:
                img = imgs_directory + key[0][0:2].lower() + '/' + key[0] + '.jpg'
            self.file_output.write(key[0])
            self.file_output.write(" ")
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

        cv2.destroyAllWindows()
        self.file_output.write("\n")
