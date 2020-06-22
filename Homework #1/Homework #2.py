# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:43:28 2018

@author: Emre
"""

import numpy as np
import random
from random import randint

#   generates N different data points
def generateData(totalPoint):
    
    dataset0 = np.ones(totalPoint)                         #   for bias value
    dataset1 = np.random.uniform(-1.0, 1.0, totalPoint)    #   x-coordinate
    dataset2 = np.random.uniform(-1.0, 1.0, totalPoint)    #   y-coordinate
    dataset = [list(a) for a in zip(dataset0, dataset1, dataset2)]
    return dataset

#   generates N different data points
def generateDataNL(totalPoint):
    
    dataset1 = np.random.uniform(-1.0, 1.0, totalPoint)    #   x1-coordinate
    dataset2 = np.random.uniform(-1.0, 1.0, totalPoint)    #   x2-coordinate
    dataset = [list(a) for a in zip(dataset1, dataset2)]
    return dataset

def lineFunction(dataset):
    #   a random data point to draw a line at D=2
    points = random.sample(dataset, 2)
    _x1 = points[0][1]
    _x2 = points[1][1]
    _y1 = points[0][2]
    _y2 = points[1][2]    
    m = abs(_y2-_y1)/abs(_x2-_x1)
    b = _y1 - m*_x1
    
    def f(x): return m*x + b
    return f

#   calculate in-sample error
def inSampleLinReg(dataset, weight, label, N):
    in_sample=0
    
    for i in range(N):
        error = ((weight[0] * dataset[i][0]) + (weight[1] * dataset[i][1]) +
                  (weight[2] * dataset[i][2]) - label[i])**2
        in_sample = in_sample + error
       
    in_sample = in_sample/N
    return in_sample

#   calculate in-sample error
def inSampleLinRegNL(dataset, weight, label, N):
    t=0
    g = np.dot(dataset, weight)
    g = np.sign(g)
    in_sample = np.subtract(g, label)
    
    for i in range(N):
        if(in_sample[i] != 0):
            t = t + 1
    
    return t/N

#   calculate out-of-sample error
def outSampleLinReg(weight):
    testData = generateData(1000)
    out_sample=0
    f = lineFunction(testData)
    
    for i in range(1000):
        y = f(testData[i][1])
        error = ((weight[0] * testData[i][0]) + (weight[1] * testData[i][1]) - y)**2
        out_sample = out_sample + error
            
    out_sample = out_sample / 1000
    return out_sample

#   test the trained PLA
def myPLATest(w, points, N):
    
    e=0
    #   generate test data
    testData = generateData(100)
    
    #   CHECK ALL TEST DATA POINTS
    for i in range(len(testData)):
           
       #    get a data point
       _bias = testData[i][0]
       _x = testData[i][1]
       _y = testData[i][2]
       
       #   THE TARGET FUNCTION
       f_x = (_x*points[1][2])-(_x*points[0][2])-(points[1][1]*points[1][2])
       +(points[1][1]*points[0][2])-(_y*points[1][1])+(points[1][1]*points[1][2])
       +(points[0][1]*_y)-(points[0][1]*points[1][2])
       
       #    Label signs
       if(f_x <= 0):
           f_x = 1
       else:
           f_x = -1

       g_x = w[0]*_bias + w[1]*_x + w[2]*_y
       
       #    Label signs
       if(g_x >= 0):
           g_x = 1
       else:
           g_x = -1
       
        #   THE DISAGREEMENT BETWEEN f(x) & g(x)
       if(f_x != g_x):
           e += 1
           
    #   take average error
    e = e / len(testData)
    return e

#   PLA algorithm runs inside
def myPLA(N):
    #   generate dataset
    trainData = generateData(N)
    #   count total number of iterations
    iteration=0
    #   initialize weight vector with zeros
    w = np.zeros(3)
    #   a random data point to draw a line at D=2
    points = random.sample(trainData, 2)
    #   check whether all dataset converged or not
    converged = False
    
    while not converged:
        
        converged = True
        #   CHECK ALL DATA POINTS
        for i in range(len(trainData)):
           
           #    get a data point
           _bias = trainData[i][0]
           _x = trainData[i][1]
           _y = trainData[i][2]
           
           #   THE TARGET FUNCTION
           f_x = (_x*points[1][2])-(_x*points[0][2])-(points[1][1]*points[1][2])
           +(points[1][1]*points[0][2])-(_y*points[1][1])+(points[1][1]*points[1][2])
           +(points[0][1]*_y)-(points[0][1]*points[1][2])
           
           #    Label zeros as ones
           if(f_x <= 0):
               f_x = 1
           else:
               f_x = -1
           
           if(f_x * (w[0]*_bias+w[1]*_x+w[2]*_y) <= 0):
               #    WEIGHT UPDATE
               w[0] += (f_x * _bias)
               w[1] += (f_x * _x) 
               w[2] += (f_x * _y)
               converged = False
               iteration += 1
               
    e = myPLATest(w, points, N)    
    return iteration, e, w

def myLinearRegression(N):
    #   generate dataset
    trainData = generateData(N)
    #   get line function
    f = lineFunction(trainData)
    #   initialize weight vector with zeros
    w = np.zeros(3)
    #   initialize target vector with zeros
    y = np.empty(N)
    
    for i in range(len(trainData)):
        #   target function value
        y[i] = f(trainData[i][1])
               

    dagger = np.linalg.pinv(trainData)
    w = np.dot(dagger, y)
    
    e_in = inSampleLinReg(trainData, w, y, N)
    e_out = outSampleLinReg(w)
    
    return e_in, e_out

def myNonLinearTransformation(N):
    #   generate dataset
    trainData = generateDataNL(N)
    #   initialize target vector with zeros
    y = np.empty(N)
    
    #   fill the target function vector
    for i in range(len(trainData)):
        y[i] = nF(trainData[i][0], trainData[i][1])
        
    noise(y)
    dagger = np.linalg.pinv(trainData)
    w = np.dot(dagger, y)
    
    e_in = inSampleLinRegNL(trainData, w, y, N)
    
    return e_in

def myNonLinearTransformation2(N):
    #   generate dataset
    temp = generateDataNL(N)
    trainData = np.empty((N,6))
    #   initialize target vector with zeros
    y = np.empty(N)
    
    
    #   fill the target function vector
    for i in range(len(temp)):
        y[i] = nF(temp[i][0], temp[i][1])
    
    trainData[:,0] = 1
    for i in range(N):
        trainData[i][1] = temp[i][0] #   x1
        trainData[i][2] = temp[i][1] #   x2
        trainData[i][3] = temp[i][0]*temp[i][1] #   x1*x2
        trainData[i][4] = temp[i][0]*temp[i][0] #   x1*x1
        trainData[i][5] = temp[i][1]*temp[i][1] #   x2*x2
        
    temp.clear()
    
    noise(y)
    dagger = np.linalg.pinv(trainData)
    w = np.dot(dagger, y)
    
    out_of_sample = myOutError(w, N)
    
    return w, out_of_sample

def myOutError(weight, N):
    out_of_sample=0
    N=1000
    #   generate dataset
    temp = generateDataNL(N)
    testData = np.empty((N,6))
    #   initialize target vector
    y = np.empty(N)
    
    #   fill the target function vector
    for i in range(len(temp)):
        y[i] = nF(temp[i][0], temp[i][1])
        
    #np.sign(y)
    noise(y)
    
    testData[:,0] = 1
    for i in range(N):
        testData[i][1] = temp[i][0] #   x1
        testData[i][2] = temp[i][1] #   x2
        testData[i][3] = temp[i][0]*temp[i][1] #   x1*x2
        testData[i][4] = temp[i][0]*temp[i][0] #   x1*x1
        testData[i][5] = temp[i][1]*temp[i][1] #   x2*x2
        
        t = testData[i][0]*weight[0]+testData[i][1]*weight[1]+testData[i][2]*weight[2]
        +testData[i][3]*weight[3]+testData[i][4]*weight[4]+testData[i][5]*weight[5]
        
        t=1 if t>0 else -1
        
        if(t - y[i] != 0):
            out_of_sample = out_of_sample + 1
        
    
    return out_of_sample/N

def nF(x1, x2):
    
    t=x1**2 + x2**2 - 0.6
    t=1 if t>0 else -1
    return t

def noise(y):
    
    for x in range(100):
        i = randint(0, 999)
        y[i]=-1 if y[i]==1 else 1
    
def main():
    
      totalIterations=0
      totalError=0
      totalIn=0
      totalOut=0
      totalIn_Non=0
      w = np.empty(6)
      
      for i in range(0, 1000):
          #   run the algorithm for 10 training points
          _iter, _avgError, finalWeights = myPLA(10)
          totalIterations += _iter
          totalError += _avgError
            
      print("Average Iterations of 10: ", totalIterations/1000)
      print("Average Error of 10: ", totalError/1000)
      #print(finalWeights)
       
      for i in range(0, 1000):
          #   run the algorithm for 10 training points
          _iter, _avgError, finalWeights = myPLA(100)
          totalIterations += _iter
          totalError += _avgError
             
      print("Average Iterations of 100: ", totalIterations/1000)
      print("Average Error of 100: ", totalError/1000)
      #print(finalWeights)

      for i in range(1000):
          tempIn, tempOut = myLinearRegression(100)
          totalIn = totalIn + tempIn
          totalOut = totalOut + tempOut
          tempIn = myNonLinearTransformation(1000)
          totalIn_Non = totalIn_Non + tempIn
          temp, out_of_sample = myNonLinearTransformation2(1000)
          w[0] = w[0] + temp[0]
          w[1] = w[1] + temp[1]
          w[2] = w[2] + temp[2]
          w[3] = w[3] + temp[3]
          w[4] = w[4] + temp[4]
          w[5] = w[5] + temp[5]
          totalError = totalError + out_of_sample
    
      print("In-sample error is: ", totalIn/1000)
      print("Out-of-sample error is: ", totalOut/1000)
      print("In-sample error is: ", totalIn_Non/1000)
      print("Average out-of-sample error is: ", totalError/1000)
      print("w[0]: ", w[0]/1000, "\nw[1]: ", w[1]/1000, "\nw[2]: ", w[2]/1000,
            "\nw[3]: ", w[3]/1000, "\nw[4]: ", w[4]/1000, "\nw[5]: ", w[5]/1000)
      
    
main()