# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:39:28 2019

@author: Emre
"""

def one_vs_one(n1, n2):
    _x = []
    _y = []
    for i in range(len(y)):
        if(y[i] == n1):
            _y.append(1)
            _x.append(x[i])
        elif(y[i] == n2):
            _y.append(-1)
            _x.append(x[i])
# =============================================================================
#         else:
#             _y[i] = 0
# =============================================================================
            
    return _y, _x

            
def one_vs_all(n1):
    _y = list(y)
    for i in range(len(_y)):
        if(y[i] == n1):
            _y[i] = 1
        else:
            _y[i] = -1
            
    return _y

def one_vs_one_T(n1, n2):
    _x = []
    _y = []
    for i in range(len(y_test)):
        if(y_test[i] == n1):
            _y.append(1)
            _x.append(x_test[i])
        elif(y_test[i] == n2):
            _y.append(-1)
            _x.append(x_test[i])
# =============================================================================
#         else:
#             _y[i] = 0
# =============================================================================
            
    return _y, _x

#import sklearn.svm.libsvm as svm
from svmutil import *

y = []
x = []

y_test = []
x_test = []

with open("train.txt", "r") as file:
    for line in file:
        buffer = line.split()
        y.append(int(float(buffer[0])))
        x1 = (float(buffer[1]))
        x2 = (float(buffer[2]))
        x.append([x1,x2])
        
with open("test.txt", "r") as file:
    for line in file:
        buffer = line.split()
        y_test.append(int(float(buffer[0])))
        x1 = (float(buffer[1]))
        x2 = (float(buffer[2]))
        x_test.append([x1,x2])
        

py = one_vs_all(0)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("0 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)
print("Total number of support vector machines: ", model.get_nr_sv())

py = one_vs_all(2)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("2 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

py = one_vs_all(4)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("4 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

py = one_vs_all(6)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("6 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

py = one_vs_all(8)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("8 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

print("==================================================")
print("==================================================")
print("==================================================")

py = one_vs_all(1)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("1 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)
print("Total number of support vector machines: ", model.get_nr_sv())

py = one_vs_all(3)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("3 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

py = one_vs_all(5)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("5 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

py = one_vs_all(7)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("7 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

py = one_vs_all(9)        
problem = svm_problem(py,x)
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
print("9 versus all: ")
p_labs, p_acc, p_vals = svm_predict(py, x, model)

print("==================================================")
print("==================================================")
print("==================================================")

py, px = one_vs_one(1,5)
pyt, pxt = one_vs_one_T(1,5)
problem = svm_problem(py, px)
problemTest = svm_problem(pyt, pxt)

print("Model accuracy with C = 0.001")
model = svm_train(problem, '-c 0.001 -t 1 -d 2') # C=0.001 Q=2
p_labs, p_acc, p_vals = svm_predict(py, px, model)
model = svm_train(problemTest, '-c 0.001 -t 1 -d 2') # C=0.001 Q=2
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("Total number of support vector machines: ", model.get_nr_sv())
print("==================================================")
print("Model accuracy with C = 0.01")
model = svm_train(problem, '-c 0.01 -t 1 -d 2') # C=0.01 Q=2
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("Total number of support vector machines: ", model.get_nr_sv())
print("==================================================")
print("Model accuracy with C = 0.1")
model = svm_train(problem, '-c 0.1 -t 1 -d 2') # C=0.1 Q=2
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("Total number of support vector machines: ", model.get_nr_sv())
print("==================================================")
print("Model accuracy with C = 1")
model = svm_train(problem, '-c 1 -t 1 -d 2') # C=1 Q=2
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("Total number of support vector machines: ", model.get_nr_sv())

print("==================================================")
print("==================================================")
print("==================================================")

print("Model accuracy with C = 0.0001")
model = svm_train(problem, '-c 0.0001 -t 1 -d 5') # C=0.001 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
print("Total number of support vector machines: ", model.get_nr_sv())
print("==================================================")
print("Model accuracy with C = 0.001")
model = svm_train(problem, '-c 0.001 -t 1 -d 5') # C=0.01 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
print("Total number of support vector machines: ", model.get_nr_sv())
print("==================================================")
print("Model accuracy with C = 0.01")
model = svm_train(problem, '-c 0.01 -t 1 -d 5') # C=0.1 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
print("Total number of support vector machines: ", model.get_nr_sv())
print("==================================================")
print("Model accuracy with C = 1")
model = svm_train(problem, '-c 1 -t 1 -d 5') # C=1 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("Total number of support vector machines: ", model.get_nr_sv())

print("==================================================")
print("==================================================")
print("==================================================")

min = 0.0
print("Ecv for C = 0.0001:")
for i in range(100):
    model = svm_train(problem, '-c 0.0001 -t 1 -d 2 -v 10') # C=0.0001 Q=2
    if(p_acc[0] > min):
        min = p_acc[0]

print("For C = 0.0001: ", min)
print("Ecv for C = 0.001:")
min = 0.0
for i in range(100):
    model = svm_train(problem, '-c 0.001 -t 1 -d 2 -v 10') # C=0.0001 Q=2
    
print("Ecv for C = 0.01:")
min = 0.0
for i in range(100):
    model = svm_train(problem, '-c 0.01 -t 1 -d 2 -v 10') # C=0.0001 Q=2

print("Ecv for C = 0.1:")
min = 0.0
for i in range(100):
    model = svm_train(problem, '-c 0.1 -t 1 -d 2 -v 10') # C=0.0001 Q=2
    if(p_acc[0] > min):
        min = p_acc[0]

print("For C = 0.1: ", min)
print("Ecv for C = 1:")
min = 0.0
for i in range(100):
    model = svm_train(problem, '-c 1 -t 1 -d 2 -v 10') # C=0.0001 Q=2
    if(p_acc[0] > min):
        min = p_acc[0]
print("For C = 0.1: ", min)

print("==================================================")
print("==================================================")
print("==================================================")

py, px = one_vs_one(5,1)
pyt, pxt = one_vs_one_T(5,1)
problem = svm_problem(py, px)
problemTest = svm_problem(pyt, pxt)

print("Model accuracy with C = 0.01")
model = svm_train(problem, '-c 0.01 -t 2') # C=0.01 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("==================================================")
print("Model accuracy with C = 1")
model = svm_train(problem, '-c 1 -t 2') # C=1 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("==================================================")
print("Model accuracy with C = 100")
model = svm_train(problem, '-c 100 -t 2') # C=100 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("==================================================")
print("Model accuracy with C = 10000")
model = svm_train(problem, '-c 10000 -t 2') # C=10000 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)
print("==================================================")
print("Model accuracy with C = 1000000")
model = svm_train(problem, '-c 1000000 -t 2') # C=1000000 Q=5
p_labs, p_acc, p_vals = svm_predict(py, px, model)
p_labs, p_acc, p_vals = svm_predict(pyt, pxt, model)