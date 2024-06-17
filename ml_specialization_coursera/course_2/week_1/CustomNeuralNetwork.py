
import numpy as np


class CustomNeuralNetwork: 

    """ 
        Do the same as the Sequential class in keras
    """
    layers = []
    def __init(self):
        pass 
        

    def addLayer(self,layer): 
        self.layers.append(layer)
        return self
    


    def predict(self,x):
        for layer in self.layers:
            print(x)
            x=layer(x)
            print(x)
        if(x>0.5):
            print('The model predicts that the house will be sold')
        else:
            print('The model predicts that the house will not be sold')

    