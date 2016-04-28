__author__ = 'david_torrejon'
import numpy as np
import os


class Img_Tbir:

        def __init__(self, filepath = './data_6stdpt/Features/scaleconcept16_ImgID.txt_Mod', is_dev=False, is_train=False, load_dict=True):
            self.all_relu_imgs, matrix_img = self.open_file_img(filepath)
            print matrix_img.size


        def open_file_img(self, filename):
            img_dict = {}
            list_to_np = []
            if os.path.isfile(filename):
                print 'found', filename, 'in directory...'
                img_found = 0
                bool_imgid = True
                with open(filename) as f:
                    print f.read()
                    while True:
                        tmp = []
                        values = []
                        c = f.read(1)
                        #print c
                        if c is not '\n' or c is not ' ':
                            tmp.append(c)
                        else:
                            if bool_imgid and c is '\n':
                                bool_imgid = False
                                img_dict[''.join(tmp)] = img_found
                            else:
                                if c is ' ':
                                    values.append(int(''.join(tmp)))
                                if c is '\n':
                                    bool_imgid = True
                                    list_to_np.append(values)
                                    print list_to_np
                                    raise SystemExit(1)
                        if not c:
                          print "End of file"
                          break

            return img_dict, np.matrix(list_to_np)

        def image_similarity(img1):
            None
