import torch
import torch.nn as nn

"""
H linear + Relu
T linear + Sigmoid
"""


"""
        etant donne que nous utilisons une fonction Relu nous allons devoir faire une 
        initialisation de Glorot de poids de notre fonction lineaire 
        
        for-each input-hidden weight
            variance = 2.0 / (fan-in +fan-out)
            stddev = sqrt(variance)
            weight = gaussian(mean=0.0, stddev)
        end-for
        
"""
        


class Plain_Net(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(Plain_Net,self).__init__()
        
        self.linear1=nn.Linear(input_shape,output_shape)
        nn.init.xavier_uniform(self.linear1.weight)
        self.activation_H=torch.nn.ReLU()
        
        
    def forward(self,x):
        
        return self.activation_H( self.linear1(x) )
        
        
        
class Highway_Net(nn.Module):
    def __init__(self,input_shape,output_shape,initial_bias=-3):
        super(Highway_Net,self).__init__()
        #H 
        self.linear1=nn.Linear(input_shape,output_shape)
        nn.init.xavier_uniform(self.linear1.weight)
        self.activation_H=torch.nn.ReLU()
        
        #T
        self.tr_gate=nn.Linear(input_shape,output_shape)
        self.activation_T=torch.nn.Sigmoid()
        self.tr_gate.bias.data.fill_(initial_bias)
        
    def forward(self,x):
        h=self.activation_H(self.linear1(x))
        t=self.activation_T(self.tr_gate(x))
        y= ( h*t ) + ( x * (1-t) )
        return y