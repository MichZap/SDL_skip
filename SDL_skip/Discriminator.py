# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:49:43 2024

@author: Michael
"""
import torch
import torch.nn as nn
import math
import numpy as np
from SDL import SkipBlockD, SkipBlockU
from model_utilis import IterativeSkip, generate_inverse_series

class Discriminator(nn.Module):
    def __init__(self, opt,eig_vecs,eig_vals):
        super().__init__()
        
        k = len(opt["nb_freq"])
        
        #n_w = 4 + 5(2**k-2) # 2 per skipping block + 1 per additional skip sum_{i=1}_{k}2**(i+1) + sum_{i=2}_{k}2**(i-1) use formula for power 2 series
        phi = [torch.tensor(eig_vecs[:,:opt["nb_freq"][i]],device = opt["device"],requires_grad=False) for i in range(k)]
                            
        Spectral_M_down = []
        l = k
        
        while l>0:
            Spectral_M_down.append(phi[l-1].t())
            j = l
            while l < k:
                i = l        
                l = l + j
                M = torch.matmul(phi[l-1].t(),phi[i-1])
                Spectral_M_down.append(M)
                del M
            l = int(math.floor(j/2))
        

        del phi
        torch.cuda.empty_cache()
        
        eig_vals = torch.tensor(eig_vals,device = opt["device"],requires_grad=False)

        learn = opt["learn"] if "learn" in opt else True
        n = sum([2**i for i in range(opt["depth"]+1)])
        
        if learn == False:
            learn = np.repeat([False],n)
            learn[-2**opt["depth"]:] = True
        else:
            learn = np.repeat([True],n)
                 
        
        self.SkipBlocksDown = nn.ModuleList(
            [
            SkipBlockD(
                hidden_dim = opt["hidden_dim"][i],
                conv_size = opt["conv_size"][i],
                phi = Spectral_M_down[i],
                eig_vals = eig_vals,
                prior_coef = opt["prior_coef"],
                activation = opt["activation"],
                dropout = opt["dropout"] if 'dropout' in opt else 0,
                use_activation = opt["use_activation"],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_soft = opt["use_soft"] if "use_soft" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                aw = opt["aw"] if "aw" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                n_heads = opt["n_heads"] if "n_heads" in opt else 0,
                value = opt["value"] if "value" in opt else True,
                learn = learn[i],
                use_norm_att = opt["use_norm_att"] if "use_norm_att" in opt else True,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                
            )
            for i in range(len(Spectral_M_down))]
        )


        self.enc = opt["encoder"] if "encoder" in opt else False
        self.l1 = nn.Linear(Spectral_M_down[-1].size(dim=0)*3, 1 ,bias = True) if self.enc == False else nn.Identity()
        self.activation = nn.Sigmoid() if self.enc == False else nn.Identity()

        
        del Spectral_M_down
        torch.cuda.empty_cache()
        
        
    def forward(self, x):
        
        x = IterativeSkip(x,self.SkipBlocksDown)
        x_flat = torch.flatten(x, start_dim=1)
        if self.enc == False:
            
            x = self.l1(x_flat)
            x = self.activation(x)
 

        return x, x_flat
    
class Generator(nn.Module):
    def __init__(self, opt,eig_vecs,eig_vals):
        super().__init__()
        
        k = len(opt["nb_freq"])
        
        #n_w = 4 + 5(2**k-2) # 2 per skipping block + 1 per additional skip sum_{i=1}_{k}2**(i+1) + sum_{i=2}_{k}2**(i-1) use formula for power 2 series
        phi = [torch.tensor(eig_vecs[:,:opt["nb_freq"][i]],device = opt["device"],requires_grad=False) for i in range(k)]
                            
        Spectral_M_down = []
        l = k
        
        while l>0:
            Spectral_M_down.append(phi[l-1].t())
            j = l
            while l < k:
                i = l        
                l = l + j
                M = torch.matmul(phi[l-1].t(),phi[i-1])
                Spectral_M_down.append(M)
                del M
            l = int(math.floor(j/2))
        

        inversindex = generate_inverse_series(len(Spectral_M_down)) 
        del phi
        torch.cuda.empty_cache()
        
        eig_vals = torch.tensor(eig_vals,device = opt["device"],requires_grad=False)

        learn = opt["learn"] if "learn" in opt else True
        n = sum([2**i for i in range(opt["depth"]+1)])
        
        if learn == False:
            learn = np.repeat([False],n)
            learn[-2**opt["depth"]:] = True
        else:
            learn = np.repeat([True],n)
                 

        
        self.SkipBlocksUp = nn.ModuleList(
            [
            SkipBlockU(
                hidden_dim = opt["hidden_dim"][inversindex[i]],
                conv_size = opt["conv_size"][inversindex[i]],
                phi = Spectral_M_down[inversindex[i]].t(),
                eig_vals = eig_vals,
                prior_coef = opt["prior_coef"],
                activation = opt["activation"],
                dropout = opt["dropout"] if 'dropout' in opt else 0,
                use_activation = opt["use_activation"],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_soft = opt["use_soft"] if "use_soft" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                aw = opt["aw"] if "aw" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                n_heads = opt["n_heads"] if "n_heads" in opt else 0,
                value = opt["value"] if "value" in opt else True,
                learn = learn[inversindex[i]],
                use_norm_att = opt["use_norm_att"] if "use_norm_att" in opt else True,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
            )
            for i in range(len(Spectral_M_down))]
        )
        
        del Spectral_M_down
        torch.cuda.empty_cache()



        
        
    def forward(self, x):
        
        x = IterativeSkip(x,self.SkipBlocksUp)

        return x