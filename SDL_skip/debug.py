# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 00:23:15 2024

@author: Michael
"""

import torch
import numpy as np
from SDL import SDL_skip
from SDL_simple import SDL_skip_simple
from utilis import gnn_model_summary

config = {
  "depth": 0,                   # model depth 0 pools directly to the latent size 
  "nb_freq": [4096,1024,256],   # number of eigenvectors considered, here i,m and j from illustration
  "device": "cuda",
  "learning_rate": 1e-3/2,
  "train_batch_size": 16,
  "test_batch_size": 16,
  "hidden_dim": [64]*7,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "conv_size": [1024]*7,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "prior_coef": 0.9,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
  "use_activation":0,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
  "dropout":0,              
  "activation": "ELU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU 
  "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
  "use_eigenvals":False,         # use eigenvalue encoding
  "use_norm":False,             # use LayerNorm before every block of linear layers
  "use_att":False,              # use Attention-Matrix for pooling
  "use_norm_att":False,          # in case of use_att True determine if the attentionscore gets normalized
  "sqrt":False,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
  "n_heads":0,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
  "value":False,                 # in case of multihead attention, use additional 3x3 value matrix
  "bias":False,                 # use bias for linear layers
  "p":0,                        # augmentation probability for flipping and rotation per axis
  "reset":True,                 # use axvier uniform weight initialization instead of pytorch default
  "aw":True,                    # use additional weight to scale every SkipBlock
  "epochs":10000,
  "typ": "BBF",                 # data to be used "BBF", "Coma" and "FWH"
  "shed_fac":0.75,              # learning rate scale factor on plateau, see X in paper
  "print_epoch":25,
  "shed_pat":25,                # learning rate sheduler patience, see N in paper
  "es_pat":100,                 # early stopping patience, see M in paper
  "Skip":2,                     # 0 no skip connections, 1 skip connection to end of each block, 2 skip connection to middle and end, 3 skip connection only to middle
  "learn":True,                # False disables the learnable component of all SkipBlocks apart of the deepest part of the skipping pyramid, reduces number of parameters approx by a factor fo 3
  "simple":True,                # uses simple model
}


config["nb_freq"] = [2]

PATH = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'+'\\'+str(config["nb_freq"][0])+'_SDL_skip.pt'

## folder where .pt-files will be stored 
config["root_dir"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\SDL'
## folder containing input ply-files
config["folder"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
## folder containing eigenvalues and eigenvectors
config["path_decomp"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'


if config["use_conv"]==False:
    config["use_eigenvals"]=False
#liveloss = PlotLosses()
#setup Model and Optimizer
#setup Model and Optimizer
if config["typ"] == "Coma":
    f = "ComaData_5023"
elif config["typ"] == "FWH":
    f = 11510
else:
    f = 31027 

#load eigenvectors and value
outfile_vec = config["path_decomp"] +"\\"+ str(f) +"eig_vecs.npy"
outfile_val = config["path_decomp"] +"\\"+ str(f) +"eig_vals.npy"

eig_vecs = np.load(outfile_vec)
eig_vals = np.load(outfile_val)

freq = max(config["nb_freq"])

eig_vals = eig_vals[:freq]
eig_vecs = eig_vecs[:,:freq]

simple = config["simple"] if 'simple' in config else False
if simple:
    model = SDL_skip_simple(config,eig_vecs,eig_vals)
else:
    model = SDL_skip(config,eig_vecs,eig_vals)

gnn_model_summary(model)
#print(model.state_dict())

print("Down",torch.matmul(model.SkipBlocksDown[0].l4.weight,model.SkipBlocksDown[0].l3.weight))
print("Up",torch.matmul(model.SkipBlocksUp[0].l4.weight,model.SkipBlocksUp[0].l3.weight))

model.load_state_dict(torch.load(PATH)['model_state_dict'])
#print(model.state_dict())
print("Down",torch.matmul(model.SkipBlocksDown[0].l4.weight,model.SkipBlocksDown[0].l3.weight))
print("Up",torch.matmul(model.SkipBlocksUp[0].l4.weight,model.SkipBlocksUp[0].l3.weight))
