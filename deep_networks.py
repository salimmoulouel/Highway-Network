#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:36:05 2019

@author: moulouel
"""
import torch.nn as nn
import NN
class Plain_deep(nn.Module):
    def __init__(self,input_shape,output_shape,nb_layers,hidden_shape):
        super(Plain_deep,self).__init__()
        self.couches=nn.ModuleList([NN.Plain_Net(input_shape,hidden_shape)])
        for i in range(1,nb_layers):#le premier a deja été créé
            self.couches.append(NN.Plain_Net(hidden_shape,hidden_shape))
        self.couches.append(nn.Linear(hidden_shape,output_shape))
        
    def forward(self,x):
        for couche in self.couches:
            x=couche(x)
        y=nn.functional.softmax(x)
        return y
        
class Highway_deep(nn.Module):
    def __init__(self,input_shape,output_shape,nb_layers,hidden_shape):
        super(Highway_deep,self).__init__()
        self.first=nn.Linear(input_shape,hidden_shape)
        nn.init.xavier_normal(self.first.weight)
        self.active_first=nn.ReLU()
        self.couches=nn.ModuleList([NN.Highway_Net(hidden_shape,hidden_shape)])
        for i in range(1,nb_layers):#le premier a deja été créé
            self.couches.append(NN.Highway_Net(hidden_shape,hidden_shape))
        self.couches.append(nn.Linear(hidden_shape,output_shape))
    
    def forward(self,x):
        x=self.active_first(self.first(x))
        for couche in self.couches:
            x=couche(x)
        y=nn.functional.softmax(x)
        return y
        