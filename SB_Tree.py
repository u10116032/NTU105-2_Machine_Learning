# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 08:45:42 2017

@author: halley
"""

import sys

class Ratio:
    numerator = 0
    denominator = 0
    
    def value(self):
        if self.denominator !=0 :
            value = self.numerator/self.denominator
        else:
            value=0
        return value




def findPath(num, den):    
    start = Ratio()
    start.numerator=0
    start.denominator=1
    
    
    end = Ratio()
    end.numerator = 1
    end.denominator=0
    inputRatio = Ratio()
    inputRatio.numerator = num
    inputRatio.denominator = den
    
    path=""
    median = Ratio()
    median.numerator = start.numerator + end.numerator
    median.denominator = start.denominator + end.denominator
    
    while True:
          
        if (inputRatio.denominator, inputRatio.numerator) == (start.denominator, start.numerator):
            print("0/1, path=NULL")
            break
        elif (inputRatio.denominator, inputRatio.numerator) == (end.denominator, end.numerator):
            print("1/0, path=NULL")        
            break
        elif inputRatio.value() == median.value():
            return path
        elif inputRatio.value() > median.value():
            start.denominator=median.denominator
            start.numerator=median.numerator
            
            path = path+'R'
        elif inputRatio.value() < median.value():
            end.denominator=median.denominator
            end.numerator=median.numerator
            path = path+'L'
        
       
        median.numerator = start.numerator + end.numerator
        median.denominator = start.denominator + end.denominator
            
        
        

file = open('input.txt', 'r')
ans = open('output.txt', 'w')

for line in file.readlines():
    line = line.split()
    if int(line[0])==int(line[1])==0:
        break
    else:
        ans.write(findPath(int(line[0]), int(line[1])))
        ans.write('\n')

ans.close()



