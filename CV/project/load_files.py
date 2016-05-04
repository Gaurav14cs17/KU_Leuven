__author__ = 'david_torrejon & Bharath Venkatesh'

import os
import numpy as np

def load_landmarks(path='./Project Data(2)/_Data/Landmarks/original/'):
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

    landmarks = []
    try:
        if os.path.isdir(path):
            for filename in os.listdir(path):
                filepath = path+filename
                with open(filepath) as fd:
                    landmarks_file = []
                    for i, line in enumerate(fd):
                        if i%2==0:
                            x = float(line)
                        else:
                            y = float(line)
                            landmarks_file.append([x,y])
                    landmarks.append(np.asarray(landmarks_file))
    except:
        print "filepath not found"

    return np.asarray(landmarks)
