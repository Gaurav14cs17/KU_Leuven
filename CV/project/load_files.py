__author__ = 'david_torrejon & Bharath Venkatesh'
import os
import numpy as np
import string

def extractLandmarksForIncisorByIndex(landmarks,index):
    shapes=[]
    for key in landmarks.keys():
        shapes.append(landmarks[key][index])
    return shapes
    
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
    ordering = {}

    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = path+filename
            parts=filename.split("-") 
            key = parts[0].translate(all_, nodigs)
            #Need to remember the file index as well
            index=parts[1].translate(all_, nodigs)
            if mirrored:
                key=str(int(key)-14)
            #print key
            if key not in landmarks.keys():
                landmarks[key] = []
                ordering[key]=[]
            #print key
            ordering[key].append(index)
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
    #Reordering each landmark list
    for key in landmarks.keys():
        #print ordering[key]
        myorder=sorted(range(len(ordering[key])), key=lambda k: ordering[key][k])
        #print myorder
        landmarks[key] = [ landmarks[key][i] for i in myorder]
    return landmarks
