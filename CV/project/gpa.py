import math
import numpy as np

def normalizeShape(l):
    '''
    Normalizes the array of landmarks and returns the norm
    '''
    norm=np.sqrt(np.sum(np.square(l),axis=0))
    return l/norm,norm

def centerShape(l):
    '''
    Centers the array of landmarks by subtracting the mean and returns the mean
    '''
    mu=numpy.mean(l,axis=0)
    return l-mu,mu

def alignLandmarks(l1,l2):
    '''
    Aligning Two CENTERED (Mean 0) Landmarks
    Returns the scaling factor s and the rotation matrix A
    such that |sAx1-x2| is minimized.
    This is implemented using the Kabasch algorithm for simplicity
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    '''
    nl1,norm1=normalizeShape(l1)
    nl2,norm1=normalizeShape(l2)
    C=np.dot(nl1.T,nl2)
    U,S,V=numpy.linalg.svd(C)
    #Wiki suggests
    #d=numpy.linalg.det(np.dot(V.T,U))
    #R=V.T*np.matrix('1 0 ; 0 d')*U.T
    A=V.T*U.T
    s= np.trace(S)*(norm2/norm1)
    return s,A
    

def gpa(X)
    '''
    Generalized Procustes Analysis, returns X and the mean with all points rotated and scaled 
    to be in the same coordinate system
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
    return Xsr,x0
