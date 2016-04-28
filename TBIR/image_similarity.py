__author__ = 'david_torrejon'
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
from sys import stdout

class Img_Tbir:

        def __init__(self, filepath = './data_6stdpt/scaleconcept16_data_visual_vgg16-relu7.dfeat', is_dev=False, is_train=False, load_dict=True):

            img_ids = self.get_img_ids(filename='./data_6stdpt/scaleconcept16.teaser.test.ImgID.txt')
            self.open_file_img(filepath, img_ids)
            #print len(self.all_relu_imgs)
            #print len(self.all_relu_imgs['7jGGwMUMtdBWdAw8'])
            #print self.all_relu_imgs['7jGGwMUMtdBWdAw8']



        def get_img_ids(self, filename):
            ids = []

            if os.path.isfile(filename):
                print 'found', filename, 'in directory...'
                with open(filename) as fd:  #filedescriptor C ftw!!!
                    ids = [line.rstrip() for line in fd]
            #print ids
            return ids

        def open_file_img(self, filename, img_test_ids):
            """
                dictrain is key = img_id value values
                dic_test is key = row_matrix value img_id
                returns, matrix nb_test * 4096
            """
            self.img_dict_train = {}
            self.img_dict_test = {}
            list_to_np = []
            if os.path.isfile(filename):
                print 'found', filename, 'in directory...'
                with open(filename) as fd:  #filedescriptor C ftw!!!
                    data=fd.readlines()
                print 'file read'
                row = 0
                for i,line in enumerate(data[1:]):
                    stdout.write("\rloading relu to dict: %d" % i)
                    stdout.flush()
                    line_split = line.rstrip().split(" ")
                    if line_split[0] in img_test_ids:
                        self.img_dict_test[row] = line_split[0]
                        row+=1
                        list_to_np.append(np.asarray(line_split[2:], dtype=float))
                    else:
                        self.img_dict_train[line_split[0]]=np.asarray(line_split[2:], dtype=float)

                """
                    for  line, img_id in zip(fd,img_ids):
                        #print line
                        img_dict[img_id] = np.asarray(line.rstrip().split(" "), dtype=float)
                """
            self.matrix_test = np.asarray(list_to_np)
            print " "
            print self.matrix_test.size
            #return img_dict_train, img_dict_test, np.asarray(list_to_np)

        def image_similarity(self, img1):
            """
                returns closest nth image to image
            """
            list_img_score = []
            closest = float('Inf')
            closest_id = ''
            value_img = self.img_dict_train[img1]
            current = euclidean_distances(self.matrix_test, value_img.reshape(1,-1))
            values_array = np.squeeze(np.asarray(current))
            #current = current.tolist()
            #print values_array
            #print np.argmax(current)
            max_indexes = values_array.argsort()[:-100][::1]
            max_list = max_indexes.tolist()
            #print len(max_indexes)
            #print max_indexes
            #current = current.tolist()
            #print 'error'
            for idx in max_list:
                #print idx
                #tuple_=[]
                print self.img_dict_test[idx], values_array[idx]
                float_val = float(values_array[idx])
                rounded_val = round(float_val, 5)
                #tuple_.append(self.img_dict_test[idx[0]])
                #tuple_.append(current[idx])
                list_img_score.append([self.img_dict_test[idx], rounded_val])

            return list_img_score
            #return max_indexes
