import numpy as np
L=[0,1,2]

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

def softmax(L):
    sum_e=0
    list_softmax =[]
    for l in L:
        sum_e+=np.exp(l)
    for l in L:
        list_softmax.append(np.exp(l)/sum_e)
    return list_softmax

print(softmax(L))

'''
import numpy as np

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())
'''