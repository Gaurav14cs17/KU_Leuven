import numpy as np
from shape_utils import *
from gpa import gpa
from pca import pcaV

class ASM:
    def __init__(self,shapes,varfrac):
        '''
        Accepts a list of shapes, and the fraction of variance
        that must be captured and fits the shape model
        by first running gpa, and then PCA on the built feature
        vector
        '''
        newshapes,_,_=gpa(shapes)
        self.X=constructMatrixFromShapes(newshapes)
        self.lambdas,self.P,self.mu=pcaV(self.X,varfrac)        
    
    def getP(self):
        '''
        Returns the matrix of eigenvectors P
        '''
        return self.P
    
    def getLambdas(self):
        '''
        Returns the vector of eigenvalues lambda
        '''
        return self.lambdas
    
    def getMeanShape(self):
        '''
        Returns the mean shape xbar
        '''
        return convertShapeVectorIntoShape(self.mu)
    
    def generateRandomShape(self):
        '''
        Returns a random n shape generated from the model for visualization
        See Section 4.1.2 of
        Cootes, Tim, E. R. Baldock, and J. Graham. "An introduction to active shape models." 
        Image processing and analysis (2000): 223-248.
        '''
        factors = np.random.uniform(-1, 1, size=self.lambdas.shape)
        b=np.multiply(factors,3*self.lambdas)
        return convertShapeVectorIntoShape(self.mu-np.dot(self.P,b))     
