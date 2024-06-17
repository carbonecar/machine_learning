
from Dense import Dense

class Sequential: 
   
    def __init(self,layers):
        self.layers=layers
        pass

    def sequential(self,x):
        dense=Dense()
        for layer in self.layers:
            out=dense.dense(x,layer.W,layer.b)
        
        return out