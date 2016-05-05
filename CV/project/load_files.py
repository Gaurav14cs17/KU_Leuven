__author__ = 'david_torrejon & Bharath Venkatesh'
import os
import numpy as np
import string

def load_landmarks(path='./Project Data(2)/_Data/Landmarks/original/',mirrored=False):
    """
    x1
    y1
    x2
    y2
    ...
    xn
    yn
    returns a numpy matrix of shape (nb_docs, nb_points, 2)
    """
    all_=string.maketrans('','')
    nodigs=all_.translate(all_, string.digits)
    landmarks = {}

    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = path+filename
            name = filename.split("-")[0]
            key = name.translate(all_, nodigs)
            if mirrored:
                key=str(int(key)-14)
            #print key
            if key not in landmarks.keys():
                landmarks[key] = []

            #print key
            with open(filepath) as fd:
                landmarks_file = []
                for i, line in enumerate(fd):
                    if i%2==0:
                        x = float(line)
                    else:
                        y = float(line)
                        landmarks_file.append([x,y])

                tmp = landmarks[key]
                #print landmarks
                landmarks[key].append(np.asarray(landmarks_file))
                #print landmarks[key]
    #print landmarks.keys()
    return landmarks
