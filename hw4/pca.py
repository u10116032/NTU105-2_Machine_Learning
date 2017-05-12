#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:50:59 2017

@author: halley
"""
#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import linalg as LA

def reconstructImg(basicNumber):
    topN_W = W[:,0:basicNumber]
    reconstruct_Imgs = []
    for pic in range(100):
        reconstruct_pic = np.zeros((4096,))
        for i in range(basicNumber):
            weight = np.inner(topN_W[:,i], X[:,pic])
            reconstruct_pic += (weight * topN_W[:,i])
        reconstruct_Imgs.append(reconstruct_pic + mean_img)
    return reconstruct_Imgs

#%%
#readfile
imgs = []
for user in ['A','B','C','D','E','F','G','H','I','J']:
    for idx in range (10):
        img = Image.open('./faceExpressionDatabase/' + user + str(idx).zfill(2) + '.bmp')
        imgs.append(np.array(img, dtype= np.float).flatten())
imgs = np.array(imgs, dtype= np.float)        
      
#mean face  
mean_img = (np.sum(imgs, axis= 0) / 100)
plt.imshow(mean_img.reshape(64,64), cmap=plt.get_cmap('gray'))

#%%
#calculate eigen vectors
X = np.transpose(imgs - mean_img)
U = np.dot(X,np.transpose(X))
eig_val, eig_vector = LA.eigh(U)
sort_idx = np.argsort(eig_val)[::-1] #sort by descending
W = eig_vector[:,sort_idx]


#%% 
#plot 9 eigenfaces       
fig = plt.figure(figsize=(3, 4))
for i in range(9):
    ax = fig.add_subplot(9/3, 3, i+1)
    ax.imshow(W[:,i].reshape(64,64), cmap=plt.get_cmap('gray'))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("{:d}st".format(i+1))
    plt.tight_layout()
    fig.suptitle('9 EigenFaces')
fig.savefig('9_eigenfaces.png')

#%%
#plot 100 original images
fig = plt.figure(figsize=(10, 15))
for i in range(100):
    ax = fig.add_subplot(100/10, 10, i+1)
    ax.imshow(imgs[i,:].reshape(64,64), cmap=plt.get_cmap('gray'))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("{:d}st".format(i+1))
    plt.tight_layout()
    fig.suptitle('100 Original Faces')
fig.savefig('100_OriginalFaces.png')

#%%
#plot 100 reconstructed images
reconstruct_pics = reconstructImg(5)
    
fig = plt.figure(figsize=(10, 15))
for i in range(100):
    ax = fig.add_subplot(100/10, 10, i+1)
    ax.imshow(reconstruct_pics[i].reshape(64,64), cmap=plt.get_cmap('gray'))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("{:d}st".format(i+1))
    plt.tight_layout()
    fig.suptitle('100 Reconstruct Faces')
fig.savefig('100_ReconstructFaces.png')

#%%
#find k (# of basis) for RMSE < 0.01
k = 0
for i in range(5,101):
    reconstruct_pics = reconstructImg(i)
    error = 0
    for idx in range(100):
        error += sum((imgs[idx,:] - np.array(reconstruct_pics[idx]))**2)
    error_rate = ((error/(100*4096))**0.5)/256
    print('error: ' + str(error_rate))
    if error_rate <=0.01:
        k=i
        print("k= "+str(k))
        break;
        
#plot 100 reconstructed images for k basis
reconstruct_pics = reconstructImg(k)
    
fig = plt.figure(figsize=(10, 15))
for i in range(100):
    ax = fig.add_subplot(100/10, 10, i+1)
    ax.imshow(reconstruct_pics[i].reshape(64,64), cmap=plt.get_cmap('gray'))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel("{:d}st".format(i+1))
    plt.tight_layout()
    fig.suptitle('100 Reconstruct Faces for {:d} components'.format(k))
fig.savefig('100_ReconstructFaces_{:d}components.png'.format(k))