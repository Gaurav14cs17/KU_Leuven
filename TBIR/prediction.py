__author__ = 'david_torrejon'
from PIL import Image
import cv2
import os.path
import operator

class TbirPredict:

    def __init__(self, dictionary):
        self.dictionary_words = dictionary

    def predict_img(self, text, nb_imgs=1):
        """
            should get the text apply something
            returns: image
        """
        imgs_directory = './data_6stdpt/Features/Visual/scaleconcepts16_images/'
        tokens = text.split(" ")
        print "predicting img for", tokens
        #image_predicted = Image.open()
        #image_predicted.show()
        #raise SystemExit(0)
        dict_img_retrieve={}
        for key, value in self.dictionary_words.items():
            curr_val = 0
            for pair in value:
                if pair[0] in tokens:
                    curr_val += int(pair[1])
                    #print pair
            #print curr_val
            if curr_val != 0:
                if len(dict_img_retrieve) < nb_imgs:
                    dict_img_retrieve[key] = curr_val
                else:
                     del dict_img_retrieve[min(dict_img_retrieve, key=dict_img_retrieve.get)]
                     dict_img_retrieve[key] = curr_val

        sorted_dict = sorted(dict_img_retrieve.items(), key=operator.itemgetter(1), reverse=True)
        #print sorted_dict
        for key in sorted_dict:
            img = imgs_directory + key[0][0:2].lower() + '/' + key[0] + '.jpg'
            print self.dictionary_words[key[0]]
            print " "
            if os.path.isdir(imgs_directory):
            #print 'dir found'
                if os.path.isfile(img):
                    #print 'img found'
                    image_predicted = cv2.imread(img)
                    cv2.imshow('img', image_predicted)
                    cv2.waitKey(0)
                else:
                    print 'image not found'
        cv2.destroyAllWindows()
