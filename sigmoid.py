import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Y=[1,1,0]
P=[0.8,0.7,0.1]

x=-20

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(x))
