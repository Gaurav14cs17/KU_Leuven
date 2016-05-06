import math
import numpy as np

def plotShapes(shapes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for points in shapes:
        xlist,ylist=zip(*(points))
        ax.plot(xlist,ylist)

def normalizeShape(l):
    '''
    Normalizes the array of landmarks and returns the norm
    '''
    norm=np.linalg.norm(l)
    return l/norm,norm

def centerShape(l):
    '''
    Centers the array of landmarks by subtracting the mean and returns the mean
    '''
    mu=np.mean(l,axis=0)
    return l-mu,mu

def centerAndScaleShape(l):
    '''
    Centers the array of landmarks by subtracting the mean and returns the mean and
    scales, returning the centered and scaled vector , the mean and the norm
    '''
    cl,mu=centerShape(l)
    ncl,norm=normalizeShape(cl)
    return ncl,mu,norm

def alignShapes(l1,l2):
    '''
    Aligning Two CENTERED (Mean 0) Landmarks
    Returns the matrix A and scaling factor s and the rotated and scaled matrix sAx1 that minimizes |sx1A-x2|
    See Appendix D of 
    Cootes, Tim, E. R. Baldock, and J. Graham. "An introduction to active shape models." 
    Image processing and analysis (2000): 223-248.
    
    This is implemented using the Kabasch algorithm for simplicity
    https://en.wikipedia.org/wiki/Kabsch_algorithm

    '''
    C=np.dot(l1.T,l2)
    U,S,V=np.linalg.svd(C)
    A=np.dot(U,V)
    s=S.sum()
    return A,s,s*np.dot(l1,A)
    

def gpa(X):
    '''
    Generalized Procustes Analysis, returns X and the mean with all points rotated and scaled 
    to be in the same coordinate system
    See Appendix A of 
    Cootes, Tim, E. R. Baldock, and J. Graham. "An introduction to active shape models." 
    Image processing and analysis (2000): 223-248.
      
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
    return Xsr,mu
