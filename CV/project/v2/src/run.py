
# coding: utf-8

# # H02A5A Computer Vision Project - Incisor Segmentation
# 
# ## Imports

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
import cv2
from incisorseg.dataset import Dataset
from incisorseg.utils import *
from asm.shape import Shape
from asm.shapemodel import ActiveShapeModel


# ## Reading the dataset

# In[2]:

data = Dataset('/home/bharath/workspace/CV/Project/data/')


# In[6]:

#img,mimg = data.get_training_images([0])
#l,ml = data.get_training_image_landmarks([0],Dataset.ALL_TEETH)
#lc,mlc = data.get_training_image_landmarks([0],Dataset.ALL_TEETH,True)
#plot_shapes(lc)
#imshow2(overlay_shapes_on_image(img[0],lc))
#plot_shapes(mlc)
#imshow2(overlay_shapes_on_image(mimg[0],mlc))
lc,mlc = data.get_training_image_landmarks(Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH,True)
landmarks = lc + mlc
model = ActiveShapeModel(landmarks)
plot_shapes(model.aligned_shapes())
#plot_shapes([model.mean_shape()])
plot_shapes(model.mode_shapes(1))


