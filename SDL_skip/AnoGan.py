# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:08:08 2024

@author: Michael
"""


from Discriminator import Discriminator, Generator
from train_AnoGan import train_wgangp
from train_Enc import train_encoder

import torch
import numpy as np



torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Settings
config = {
  "depth": 1,                   # model depth 0 pools directly to the latent size 
  "nb_freq": [512,170],   # number of eigenvectors considered, here i,m and j from illustration
  "device": "cuda",
  "learning_rate": 1e-4/4,
  "train_batch_size": 64,
  "test_batch_size": 64,
  "hidden_dim": [64]*7,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "conv_size": [1024]*7,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "prior_coef": 0.8,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
  "use_activation":2,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
  "dropout":0.1,              
  "activation": "ELU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU 
  "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
  "use_eigenvals":True,         # use eigenvalue encoding
  "use_norm":False,             # use LayerNorm before every block of linear layers
  "use_att":True,              # use Attention-Matrix for pooling
  "use_norm_att":True,          # in case of use_att True determine if the attentionscore gets normalized
  "sqrt":True,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
  "n_heads":0,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
  "value":False,                 # in case of multihead attention, use additional 3x3 value matrix
  "bias":False,                 # use bias for linear layers
  "p":0,                        # augmentation probability for flipping and rotation per axis
  "reset":True,                 # use axvier uniform weight initialization instead of pytorch default
  "aw":True,                    # use additional weight to scale every SkipBlock
  "epochs":1000,
  "typ": "BBF",                 # data to be used "BBF", "Coma" and "FWH"
  "Skip":2,                     # 0 no skip connections, 1 skip connection to end of each block, 2 skip connection to middle and end, 3 skip connection only to middle
  "learn":True,                # False disables the learnable component of all SkipBlocks apart of the deepest part of the skipping pyramid, reduces number of parameters approx by a factor fo 3
  "simple":False,                # uses simple model
  "direct":True,                # if True the second skip conncetion start from the input
  "n_critic":7,
  "lambda":10,
  "mean":False,
  "k": 1,
  "ada":True,
  "target_value":0.6,
  "Tau":0,
  "sym":True,
}

### folder where .pt-files will be stored 
config["root_dir"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new\pt'
#config["root_dir"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\exp_transfer\procr\pt"
## folder containing input ply-files
#config["folder"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\exp_transfer\procr"
config["folder"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new'

## folder containing eigenvalues and eigenvectors
config["path_decomp"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'
config["mat_path"] = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"

config["outpath"] =  r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\WGAN"
config["example_path"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\Data\Synthetic_3D_face_00001.ply"

f = 31027 

#load eigenvectors and value
outfile_vec = config["path_decomp"] +"\\"+ str(f) +"eig_vecs.npy"
outfile_val = config["path_decomp"] +"\\"+ str(f) +"eig_vals.npy"

eig_vecs = np.load(outfile_vec)
eig_vals = np.load(outfile_val)

freq = max(config["nb_freq"])

eig_vals = eig_vals[:freq]
eig_vecs = eig_vecs[:,:freq]

Gen = Generator(config,eig_vecs,eig_vals)
Disc = Discriminator(config,eig_vecs,eig_vals)

config["encoder"] = True
Enc =  Discriminator(config,eig_vecs,eig_vals)


train_wgangp(config, Gen, Disc,0.5)
#train_encoder(Enc, Gen, Disc, config)    




