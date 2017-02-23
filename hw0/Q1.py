# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:22:52 2017

@author: halley
"""

import numpy as np;
import sys;

fileA  = open(sys.argv[1], 'r')
fileB  = open(sys.argv[2], 'r')


matrixA = []

for index, line in enumerate(fileA.readlines()):
    matrixA.append(np.array(list(map(int, line.split(',')))))
fileA.close();

matrixB = []

for index, line in enumerate(fileB.readlines()):
    matrixB.append(np.array(list(map(int, line.split(',')))))
fileB.close();

temp=[]
matrixC = np.dot(matrixA, matrixB)
for elm in matrixC:
    temp.extend(elm)
temp.sort();

ans_file = open("ans_one.txt", 'w')
for elm in temp:
    ans_file.write(str(elm))
    ans_file.write('\n')
ans_file.close()
