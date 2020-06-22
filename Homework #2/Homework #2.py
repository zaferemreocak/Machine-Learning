# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:19:10 2018

@author: Emre
"""

from math import exp
import numpy as np
import random
import os
import sys

def myGradientDescent(u, v):
    
    learningRate = 0.1
    vt = [0] * 2
    t = 0
    
    while True:
        #   Compute the gradient
        dE = 2 * (u * exp(v) - 2 * v * exp(-u))
        dE_du = dE * (exp(v) + 2 * v * exp(-u))
        dE_dv = dE * (u * exp(v) - 2 * exp(-u))
        
        vt[0] = (-1 * dE_du)    
        vt[1] = (-1 * dE_dv)
        u = u + learningRate * vt[0]
        v = v + learningRate * vt[1]
            
        errorRate = (u * exp(v) - 2 * v * exp(-u)) ** 2
            
            
            
        if(errorRate < 10 ** (-14)):
            return u, v, t+1
            
        #   if error does not fall below 10^14, keep going
        t = t+1
            
def myCoordinateGradientDescent(u, v):
    
    learningRate = 0.1
    vt = [0] * 2
    t = 0
    
    w = [u, v]
    
    for t in range(0,30):
        #   Compute the gradient
        dE = 2 * (w[0] * exp(w[1]) - 2 * w[1] * exp(-w[0]))
        dE_du = dE * (exp(w[1]) + 2 * w[1] * exp(-w[0]))
        dE_dv = dE * (w[0] * exp(w[1]) - 2 * exp(-w[0]))
        
        vt[0] = (-1 * dE_du)
        vt[1] = (-1 * dE_dv)
        
        if(t % 2 == 0):
            w[0] = w[0] + learningRate * vt[0]
        else:
            w[1] = w[1] + learningRate * vt[1]
            
        errorRate = (w[0] * exp(w[1]) - 2 * w[1] * exp(-w[0])) ** 2
        
    return errorRate

def generateData():
    
    dataset0 = np.ones(100)                         #   bias
    dataset1 = np.random.uniform(-1.0, 1.0, 100)    #   x-coordinate
    dataset2 = np.random.uniform(-1.0, 1.0, 100)    #   y-coordinate
    dataset = [list(a) for a in zip(dataset0, dataset1, dataset2)]
    return dataset
    
def testG(w, points):
    #   generates 100 uniformly distributed random data for testing
    testData = generateData()
    eOut = 0
    #print(w[0], w[1], "points: ", points)
    for i in range(0, 100):
           #    traverse each data point
           _bias = testData[i][0]
           _x = testData[i][1]
           _y = testData[i][2]
           
           #    find out the output label for the corresponding data point
           label = (_x*points[1][2])-(_x*points[0][2])-(points[1][1]*points[1][2])
           +(points[1][1]*points[0][2])-(_y*points[1][1])+(points[1][1]*points[1][2])
           +(points[0][1]*_y)-(points[0][1]*points[1][2])
           
           if(label <= 0):
               _out=1
           else:
               _out=-1
                  
           eOut += np.log(1 + exp(-1 * _out * ((w[0]*_bias)+(w[1]*_x)+(w[2]*_y))))
    
    #    calculate E_in (cross-entropy error)
    eOut = eOut/100
    return eOut

def myLogisticRegression():
    
    #   initialize learning rate
    learningRate = 0.01
    #   generates 100 uniformly distributed random data for training
    trainData = generateData()
    #   initialize weights
    w = np.zeros(3)
    #   a random data point to draw a line at D=2
    points = random.sample(trainData, 2)
    #   initialize epoch=0, error and 2 variable to hold old weights
    epoch=0
    error=0 
    oldBias=0
    oldX=0
    oldY=0
    
    #   EACH RUN
    while True:
        
        #   shuffle the dataset
        random.shuffle(trainData)
        
        w[0] = oldBias
        w[1] = oldX
        w[2] = oldY
        
        #   EACH EPOCH
        for i in range(0, 100):
           
           #    traverse each data point
           _bias = trainData[i][0]
           _x = trainData[i][1]
           _y = trainData[i][2]
           #    find out the output label for the corresponding data point
           label = (_x*points[1][2])-(_x*points[0][2])-(points[1][1]*points[1][2])
           +(points[1][1]*points[0][2])-(_y*points[1][1])+(points[1][1]*points[1][2])
           +(points[0][1]*_y)-(points[0][1]*points[1][2])
           
           if(label <= 0):
               _out=1
           else:
               _out=-1
           
           g = 1 + exp(_out * ((w[0]*_bias)+(w[1]*_x)+(w[2]*_y)))
           g_bias = (-1 * _out * _bias) / g
           g_x = (-1 * _out * _x) / g
           g_y = (-1 * _out * _y) / g
           
           #    update the weights
           w[0] = w[0] - learningRate * g_bias
           w[1] = w[1] - learningRate * g_x
           w[2] = w[2] - learningRate * g_y
              
           
        #    increase the total number of epochs
        epoch = epoch+1
        #    calculate the stop criteria
        stopCondition = np.sqrt((oldBias-w[0])**2 + (oldX-w[1])**2 + (oldY-w[2])**2)
        #    check stop criteria
        if(stopCondition < 0.01):
            break
        else:
            #    holds predecessor weight values
            oldBias = w[0]
            oldX = w[1]
            oldY = w[2]
    
    #   find the out-of-sample error
    error = testG(w, points)
    #print("Eout is: ", error, "Total # of epoch: ", epoch)
    return error, epoch
    
def myRegüle():
    
    b=[]
    x1=[]
    x2=[]
    x1_2=[]
    x2_2=[]
    x1_x2=[]
    absx1x2sub=[]
    absx1x2add=[]
    y=[]
    
    #   IN_SAMPLE
    with open(os.path.join(sys.path[0], "in.txt"), "r") as f:
        buffer = []
        for line in f:
            buffer = line.split()
            
            _x1=float(buffer[0])
            _x2=float(buffer[1])
            
            b.append(1)
            x1.append(_x1)
            x2.append(_x2)
            x1_2.append(_x1*_x1)
            x2_2.append(_x2*_x2)
            x1_x2.append(_x1*_x2)
            absx1x2sub.append(abs(_x1-_x2))
            absx1x2add.append(abs(_x1+_x2))
            
            y.append(float(buffer[2]))
            
    
    dataset = np.column_stack((b,x1,x2,x1_2,x2_2,x1_x2,absx1x2sub,absx1x2add))
    y = np.column_stack((y))
    
    w = np.linalg.pinv(dataset)
    w = np.dot(y, w.T)
    
    g = np.sign(np.dot(w, dataset.T)) - y
    print("In-sample error: ", np.count_nonzero(g) / len(g.T), "''w/o regularization''")
    
    #   WEIGHT DECAY
    lambda_reg = 10**-3
    
    w_reg = np.linalg.inv(np.dot(dataset.T,dataset) + lambda_reg * np.identity(len(np.dot(dataset.T,dataset))))
    w_reg = np.dot(w_reg, dataset.T)
    w_reg = np.dot(w_reg, y.T)
    yhat_in = np.sign(np.dot(dataset, w_reg)) - y.T
    
    print("In-sample error (regularized): ", np.count_nonzero(yhat_in) / len(yhat_in))
    
    b.clear()
    x1.clear()
    x2.clear()
    x1_2.clear()
    x2_2.clear()
    x1_x2.clear()
    absx1x2sub.clear()
    absx1x2add.clear()
    
    dataset = np.empty(0)
    g = []
    w = []
    y = []
    
    #   OUT_OF_SAMPLE
    with open(os.path.join(sys.path[0], "out.txt"), "r") as f:
        buffer = []
        for line in f:
            buffer = line.split()
            
            _x1=float(buffer[0])
            _x2=float(buffer[1])
            
            b.append(1)
            x1.append(_x1)
            x2.append(_x2)
            x1_2.append(_x1*_x1)
            x2_2.append(_x2*_x2)
            x1_x2.append(_x1*_x2)
            absx1x2sub.append(abs(_x1-_x2))
            absx1x2add.append(abs(_x1+_x2))
            
            y.append(float(buffer[2]))
    
    dataset = np.column_stack((b,x1,x2,x1_2,x2_2,x1_x2,absx1x2sub,absx1x2add))
    y = np.column_stack((y))
    
    w = np.linalg.pinv(dataset)
    w = np.dot(y, w.T)
    
    g = np.sign(np.dot(w, dataset.T)) - y
    print("Out-of-sample error: ", np.count_nonzero(g) / len(g.T), "''w/o regularization''")
    
    #   WEIGHT DECAY
    lambda_reg = 10**-3
    
    w_reg = np.linalg.inv(np.dot(dataset.T,dataset) + lambda_reg * np.identity(len(np.dot(dataset.T,dataset))))
    w_reg = np.dot(w_reg, dataset.T)
    w_reg = np.dot(w_reg, y.T)
    yhat_out = np.sign(np.dot(dataset, w_reg)) - y.T
    print("Out-of-sample error (regularized): ", np.count_nonzero(yhat_out) / len(yhat_out))
    
    b.clear()
    x1.clear()
    x2.clear()
    x1_2.clear()
    x2_2.clear()
    x1_x2.clear()
    absx1x2sub.clear()
    absx1x2add.clear()
          
def main():
    avgError=0
    avgEpoch=0
    #   Question 5 & 6 of HOMEWORK #5
    final_U, final_V, totalIterations = myGradientDescent(1,1)
    print("##############################")
    print("QUESTION 5 & 6")
    #print total iterations
    print("Total Iterations: ", totalIterations)
    #print u and v weights
    print("u: ", final_U, " ", "v: ", final_V)
    #####
    print("##############################")
    #####
    #   Question 7 of HOMEWORK #5
    print("QUESTION 7")
    coordinateErrorRate = myCoordinateGradientDescent(1,1)
    #print final error rate after function's work is done
    print("The error rate after 30 iterations: ", coordinateErrorRate)
    #####
    print("##############################")
    #####
    #   Question 8 of HOMEWORK #5
    print("QUESTION 8 & 9")
    for i in range(0 ,100):
        r, e = myLogisticRegression()
        avgError += r
        avgEpoch += e
    avgError=avgError/100
    avgEpoch=avgEpoch/100
    print("Average Error: ", avgError, "Average Epoch: ", avgEpoch)
    #####
    print("##############################")
    #####
    #   Question 2 - 6 of HOMEWORK #6
    print("QUESTION 8 & 9")
    myRegüle()
    #####
    print("##############################")
    #####
main()
