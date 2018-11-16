import numpy as np
Y=[1,1,0]
P=[0.8,0.7,0.1]
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
'''
def cross_entropy(Y, P):
    ce = []
    for y, p in zip(Y, P):
        if y == 1:
            ce.append(np.log(p))
        else:
            ce.append(np.log(1-p))
    return sum(ce)*-1
print(cross_entropy(Y,P))
'''
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

print(cross_entropy(Y,P))