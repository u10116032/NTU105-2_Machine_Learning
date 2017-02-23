# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:52:30 2017

@author: halley
"""

from PIL import Image
import numpy as np
import sys

imgA = np.asarray(Image.open(sys.argv[1]))
imgB = np.asarray(Image.open(sys.argv[2]))
imgC = np.zeros(imgA.shape, dtype=np.uint8)
for (row, col, pValue), pixelA in np.ndenumerate(imgA):
    if  np.array_equal(imgA[row][col],imgB[row][col]):
        imgC[row][col] = [255,255,255,0]
    else:
        imgC[row][col] = imgB[row][col]

ans = Image.fromarray(imgC, mode='RGBA')
ans.save('ans_two.png')
        
