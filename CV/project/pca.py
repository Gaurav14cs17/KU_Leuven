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
    See Appendix E of 
    Cootes, Tim, E. R. Baldock, and J. Graham. "An introduction to active shape models." 
    Image processing and analysis (2000): 223-248.
    '''
    [n,d] = X.shape
    if n>d:
        C = np.dot(X.T,X)
        [l,W] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [l,eigenvectors] = np.linalg.eigh(C)
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
    return lall[1:k],Wall[:,1:k],mu
    
def pcaV(X,varianceFraction=0.9):
    '''
    Perform PCA on X and return the eigenvalues,eigenvectors and the average point
    such that varianceFraction is the fraction of the total variance captured
    '''
    mu = X.mean(axis=0)
    lall,Wall=eigop(X - mu)
    varfrac = np.cumsum(lall/np.sum(lall))
    k=bisect.bisect_right(varfrac,varianceFraction)
    return lall[1:k],Wall[:,1:k],mu
