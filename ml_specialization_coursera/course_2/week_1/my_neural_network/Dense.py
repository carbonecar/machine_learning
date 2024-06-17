import numpy as np

class Dense: 
    def __init__(self):
        pass

    def dense(self,a_in,W,b):
        units=W.shape[1] # number of units in the layer
        a_out=np.zeros(units)
        for j in range(units):
            w=W[:,j]
            z=np.dot(w,a_in)+b[j]
            a_out[j]=self.sigmoid(z)
        return a_out
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def dense_efficet(self,AT,W,B):
        Z=np.matmul(AT,W)+B
        A_out=self.sigmoid(Z)
        return A_out
        