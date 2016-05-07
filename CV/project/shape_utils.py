import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convertShapeVectorIntoShape(shapeVector):
    return(np.reshape(shapeVector,(len(shapeVector)/2,2)).tolist())
    
def collapseShapeIntoList(shape):
    return np.reshape(np.array(shape),2*len(shape)).tolist()

def constructMatrixFromShapes(shapes):
    xvectors=[]
    for shape in shapes:
        xvectors.append(collapseShapeIntoList(shape))
    return np.array(xvectors)

def plotShapes(shapes):
    '''
    Utility to plot a list of shapes
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for points in shapes:
        xlist,ylist=zip(*(points))
        ax.plot(xlist,ylist)
    plt.show()

def normalizeShape(l):
    '''
    Normalizes the shape and also returns the norm
    '''
    norm=np.linalg.norm(l)
    return l/norm,norm

def centerShape(l):
    '''
    Centers the shape by subtracting the mean and  also returns the mean
    '''
    mu=np.mean(l,axis=0)
    return l-mu,mu

def centerAndScaleShape(l):
    '''
    Centers the shape and scales it calling center and normalize
    also returns the mean and the norm
    '''
    cl,mu=centerShape(l)
    ncl,norm=normalizeShape(cl)
    return ncl,mu,norm
