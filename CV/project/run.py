__author__ = 'david_torrejon & Bharath Venkatesh'


from load_files import load_landmarks,extractLandmarksForIncisorByIndex
from load_images import load_images
from shape_utils import *
from asm import ASM

"""
Schedule:
wed:
  @david: read files and load it x = x + Pb
  numpy array
thurs:
 @bharath: move pca coord into this xD PCA PCA
We should have this by next wendsday
"""

lpath='/home/bharath/workspace/CV/Project/data/Landmarks/original/'
ipath='/home/bharath/workspace/CV/Project/data/Radiographs/'
landmarks=load_landmarks(path=lpath)
images,data=load_images(landmarks,path=ipath)

#show(images['1'])
#for index in range(8):
#    shapes=extractLandmarksForIncisorByIndex(landmarks,index)
#    centeredShapes = []
#    for shape in shapes:
#        cshape,mu=centerShape(shape)
#        centeredShapes.append(cshape)
#    plotShapes(centeredShapes)
#shapes=extractLandmarksForIncisorByIndex(landmarks,3)
#cshapes=[]
#for shape in shapes:
#    cshape,_=centerShape(shape)
#    cshapes.append(cshape)
#plotShapes(cshapes)
#ashape,_=alignAtoBProcrustes(cshapes[6],cshapes[1])
#plotShapes([cshapes[6],cshapes[1]])
#plotShapes([ashape,cshapes[1]])
#newshapes,meanShape,t=gpa(shapes)
#plotShapes([meanShape])
#plotShapes(newshapes)
#ashapemat=[]
#ashapemean=[]
for i in range(8):
    shapes=extractLandmarksForIncisorByIndex(landmarks,i)
    #newshapes,meanShape,t=gpa(shapes)
    #print 'Completed gpa of incisor ' + str(i) + ' in ' + str(t) + ' iterations '
    #plotShapes([meanShape])
    #ashapemat.append(newshapes)
    #ashapemean.append(meanShape)
    #X=prePCA(shapes)
    #l,W,mu=pcaV(X,0.9)
    #print W.shape
    #print l
    model = ASM(shapes,0.9)
    print model.getLambdas()
    plotShapes([model.getMeanShape(),model.generateRandomShape(),model.generateRandomShape(),model.generateRandomShape()])   
    
