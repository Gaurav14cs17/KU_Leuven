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

def alignAtoBProcrustes(A,B):
    '''
    Compute parameters to optimally align Two CENTERED (Mean 0) and SCALED (Unit norm) shapes A,B
    Returns rotated and scaled matrix Rx1 that minimizes |x1R-x2| (least squares)
    See Appendix D of 
    Cootes, Tim, E. R. Baldock, and J. Graham. "An introduction to active shape models." 
    Image processing and analysis (2000): 223-248.
    
    Derivation T- transpose I - inverse PI -pseudoinverse
    x1*R=x2
    T(x2)*x1*R=T(x2)*x2 
    Let C = T(x2)*x1
    R=I(C)T(x2)*x2
    For least squares fit find pseudoinverse of I using svd write C=U*S*T(V) then
    PI(C)=V*I(S)*T(U)
    R=V*I(S)*T(U)

    '''
    C=np.dot(B.T,A)
    BTB=np.dot(B.T,B)
    U,s,V=np.linalg.svd(C)
    Sinv=np.diag(np.reciprocal(s))
    PIC=np.dot(np.dot(V.T,Sinv),U.T)
    R=np.dot(PIC,BTB)
    return np.dot(A,R),R 

def gpa(shapes,tol=1e-5):
    '''
    Generalized Procustes Analysis, returns the rotated scales and shapes and the mean with all points rotated and scaled 
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
    X=np.array(shapes)
    [N,l,u]=X.shape
    for i in range(N):
        X[i,:,:],_=centerShape(X[i,:,:])
    [N,l,u]=X.shape
    currmux0=X[0,:,:]
    lastmux0 = currmux0
    t=0
    while(t==0 or np.linalg.norm(currmux0-lastmux0) > tol):
        for i in range(N):
            cshape,_=centerShape(X[i,:,:])
            X[i,:,:],_=alignAtoBProcrustes(cshape,currmux0)
        lastmux0=currmux0
        currmux0,_,_=centerAndScaleShape(np.mean(X,axis=0))
        t = t+1
    return X.tolist(),currmux0,t
