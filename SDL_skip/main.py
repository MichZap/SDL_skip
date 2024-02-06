import os
import torch
import numpy as np
import random
import pandas as pd
from train import train_bf




torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Settings
config = {
  "depth": 2,                   # model depth 0 pools directly to the latent size 
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





## folder where .pt-files will be stored 
config["root_dir"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\SDL'
## folder containing input ply-files
config["folder"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
## folder containing eigenvalues and eigenvectors
config["path_decomp"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'


i=0
result_df = pd.DataFrame(columns=["nb_freq","epoch","best_epoch","MAE_VAL","EUC_VAL","MAE_TRAIN","EUC_TRAIN"]) 
for f in [2,3,5,6,10,11,21,42,85,170,341,682,1365]:
    np.random.seed(31415)
    torch.manual_seed(31415)
    random.seed(31415)
    os.environ['PYTHONHASHSEED'] = str(31415)
    if config["depth"] == 0:
        config["nb_freq"] = [f]
    else:
        config["nb_freq"].append(f)
    results = train_bf(config)
    result_df.loc[i]=results
    i = i+1
    print(result_df)
    config["nb_freq"] = config["nb_freq"][:-1]

