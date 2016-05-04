__author__ = 'david_torrejon & Bharath Venkatesh'

import cv2
import sys
import os
import numpy as np




def load_images(landmarks, path='./Project Data(2)/_Data/Radiographs/'):
    """
        shows an img, and its corresponding landmarks
    """
    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = path+filename
            if os.path.isfile(filepath):
                radiography_nb = int(filename.split(".")[0])
                print radiography_nb
                landmark = landmarks[str(radiography_nb)]
                im = cv2.imread(filepath)
                for l in landmark:
                    pts = np.array(l, np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(im,[pts],True,(0,255,255))
                cv2.imshow(filename, im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
