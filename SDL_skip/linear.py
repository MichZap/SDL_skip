# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 04:37:43 2024

@author: Michael
"""
import torch
import torch.nn as nn


class linear(nn.Module):
    def __init__(self, opt,eig_vecs):
        super().__init__()

        n = eig_vecs.shape[0]
        self.n = n
        self.phi = torch.tensor(eig_vecs[:,:opt["nb_freq"][-1]],device = opt["device"],requires_grad=False)
        
        self.ld = nn.Linear(3*n, 3*opt["nb_freq"][-1],False)
        self.lu = nn.Linear(3*opt["nb_freq"][-1],3*n,False)
        self.weights = nn.Parameter(torch.Tensor(2, 1))

        nn.init.constant_(self.weights, opt["prior_coef"])

    def encoder(self,x):
        
        y = torch.matmul(self.phi.T,x)
        y = torch.flatten(y, start_dim=1)
        x = torch.flatten(x, start_dim=1)
        z = self.ld(x)
        z = self.weights[0]*y + (1-self.weights[0])*z
        return z
    
    def decoder(self,z):
        y = z.view(-1,int(z.size(1)/3),3)
        y = torch.matmul(self.phi,y)
        x = self.lu(z)
        x = x.view(-1,self.n,3)
        x = self.weights[1]*y + (1-self.weights[1])*x
        return x

        
    def forward(self, x):
        
        z = self.encoder(x)
        x = self.decoder(z)

        return x