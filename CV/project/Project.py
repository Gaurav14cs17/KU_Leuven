
# coding: utf-8

# # Computer Vision Project - Incisor Segmentation
# 
# 

# ## Jupyter Magics

# In[57]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import cv2
def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    


# ## load_landmarks.py - Load Landmarks

# In[58]:

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


# ## load_images.py - Load Images

# In[59]:

__author__ = 'david_torrejon & Bharath Venkatesh'

import cv2
import sys
import os
import numpy as np

def load_images(landmarks, path='./Project Data(2)/_Data/Radiographs/'):
    """
        shows an img, and its corresponding landmarks
    """
    img_landmark = {}
    matrix_images = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            filepath = path+filename
            if os.path.isfile(filepath):
                radiography_nb = int(filename.split(".")[0])
                #print radiography_nb
                landmark = landmarks[str(radiography_nb)]
                im = cv2.imread(filepath)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                for l in landmark:
                    pts = np.array(l, np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(im,[pts],True,(0,255,255))
                img_landmark[str(radiography_nb)] = im
                #print type(im)
                #print gray.shape
                matrix_images.append(gray)
                #cv2.imshow(filename, im)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
    return img_landmark, np.asarray(matrix_images)


# ## gpa.py - Do generalized procrustes analyses
# See Appendix A - Aligning the Training Set, pg 21 of
# Cootes, Tim, E. R. Baldock, and J. Graham. "An introduction to active shape models." 
# Image processing and analysis (2000): 223-248.
# 

# In[60]:

'''
1. Translate each example so that its centre of gravity is at the origin.
2. Choose one example as an initial estimate of the mean shape and scale so that |x| = sqrt(x1^2+x2^2..)=1 
3. Record the first estimate as x0 to define the default orientation.
4. Align all the shapes with the current estimate of the mean shape. 
5. Re-estimate the mean from aligned shapes.
6. Apply constraints on scale and orientation to the current estimate of the mean by aligning it with ¯
   x0 and scaling so that |x| = 1.
7. If not converged, return to 4.
   (Convergence is declared if the estimate of the mean does not change significantly after an iteration)
'''


# ## pca.py - Do principal component analysis

# In[61]:

import numpy as np
import bisect

def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    return np.dot(X-mu,W)

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    return np.dot(W,np.transpose(Y))+mu

def eigop(X):
    '''
    Do the eigendecomposition of the matrix XX' or X'X depending on the shape of the matrix
    and return ALL the eigenvectors and eigenvalues 
    '''
    [n,d] = X.shape
    if n>d:
        C = np.dot(X.T,X)
        [l,W] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [l,W] = np.linalg.eigh(C)
        W = np.dot(X.T,eigenvectors)
        for i in xrange(n):
            W[:,i] =W[:,i]/np.linalg.norm(W[:,i])
    indices=np.argsort(l)[::-1][:n]
    l=l[indices]
    W=W[:,indices]
    return l,W
    
def pcaN(X,k):
    '''
    Perform PCA on X and return the top k eigenvalues,eigenvectors and the average point
    '''
    mu = X.mean(axis=0)
    lall,Wall=eigop(X - mu)
    return l[1:k],W[:,1:k],mu
    
def pcaV(X,varianceFraction=0.9):
    '''
    Perform PCA on X and return the eigenvalues,eigenvectors and the average point
    such that varianceFraction is the fraction of the total variance captured
    '''
    mu = X.mean(axis=0)
    lall,Wall=eigop(X - mu)
    varfrac = np.cumsum(lall/np.sum(lall))
    k=bisect.bisect_right(varfrac,varianceFraction)
    return l[1:k],W[:,1:k],mu


# ## Reading Radiograms, Segmentation and Landmarks

# In[62]:

landmarks=load_landmarks(path='/home/bharath/workspace/CV/Project/data/Landmarks/original/')
mirrored_landmarks=load_landmarks(path='/home/bharath/workspace/CV/Project/data/Landmarks/mirrored/',mirrored=True)
images,data=load_images(landmarks,path='/home/bharath/workspace/CV/Project/data/Radiographs/')
#mirrored_images,data=load_images(mirrored_landmarks,path='/home/bharath/workspace/CV/Project/data/Radiographs/')


# ## Scratch

# In[63]:

#mirrored_landmarks['1']
#landmarks['1']
#show(images['1'])
#show(mirrored_images['1'])

