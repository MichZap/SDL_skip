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
    
    #learning settings:
    "device": "cuda",
    "learning_rate": 1e-3/2,
    "train_batch_size": 16,
    "test_batch_size": 16,
    "epochs":10000,
    "lambda":0,                   # additonal loss function if FFW == True for spectral latent denoiseing
    
    #learning rate shedule settings:
    "shed_fac":0.75,              # learning rate scale factor on plateau, see X in paper
    "print_epoch":25,
    "shed_pat":100,               # learning rate sheduler patience, see N in paper
    "es_pat":300,                 # early stopping patience, see M in paper
    "min_delta":0,                # parameter for early stopping tolerance
    
    #model base settings:
    "depth": 2,                   # model depth 0 pools directly to the latent size 
    "nb_freq": [4096,1024,256,85],                # number of eigenvectors considered, here i,m and j from illustration
    "hidden_dim": [16]*8,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
    "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
    "use_eigenvals":True,         # use eigenvalue encoding
    "conv_size": [1024]*8,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
    "prior_coef": 0.8,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
    
    #model additional settings:
    "bias":False,                 # use bias for linear layers
    "dropout":0.0,
    "reset":False,                # use axvier uniform weight initialization instead of pytorch default
    "aw":True,                    # use additional weight to scale every SkipBlock
    "Skip":2,                     # 0 no skip connections, 1 skip connection to end of each block, 2 skip connection to middle and end, 3 skip connection only to middle
    "learn":True,                 # False disables the learnable component of all SkipBlocks apart of the deepest part of the skipping pyramid, reduces number of parameters approx by a factor of 3
    "direct":True,                # if True the second skip conncetion start from the input
    "FFW":False,                   # additonal FFW layer to denosie the latent space for spectral decomp   
    
    #activation settings:
    "use_activation":0,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
    "activation": "ELU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU   
        
    #attention settings:
    "use_att":False,              # use Attention-Matrix for pooling
    "n_heads":8,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
    "use_norm":False,             # use LayerNorm before every block of linear layers
    "use_norm_att":"Soft",        # in case of use_att True determine if the attentionscore gets normalized
    "sqrt":True,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
    "value":False,                # in case of multihead attention, use additional 3x3 value matrix

    #augmentation settings
    "p":0,                      # augmentation probability for flipping and rotation per axis
    "k":2,                        # max amount of consecutive augmentations
    "deg":180,                     #rotation degree for augmentation
    "flip":True,                 #False disables flipping augmentation
    "add_data":False,             # if we use addition data from folder_add, not considered in train test split
    "sym":"asym",                  # False default, "sym" adds mirrored and symmetric mesh, "aysm" additional transfers asymtries from one face to another requires indices for mirroring
    "p_asym":0.05,                 # probability for asymmetrty transfer to control ratio between real and asymmetric data

    #SDL_simple                    Similar to SDL but no with constant width and advanced aggregation by spectral dim
    "simple":False,                # uses simple model
    "gate":False,                  # for simple version gate the weight parameter if true
    "flatten":False,               # flatten of the hidden dim during pooling only work if use_att false and simpel True
    "proj":True,                   # if False uses weights instead of proj matrix in mh attention
    "Pai": True,                   # Pai conv in the end
    "paiconv_size":[9,9],          # Pai conv size
    "in_ch":[3,3],                 # Pai conv in channels
    "out_ch":[3,3],                # Pai conv out channels
    "pai_act":[1,1],               # Pai conv activation  0 or 1
    "pai_skip":[True,True],
    "pai_small":[False,False],
    
    #additonal settings:
    "typ": "BBF",                 # data to be used "BBF", "Coma" and "FWH"
    "sd":False,                   # sd normalization in Dataloader
    "SSH": False,                 # normalizetion in data loader such that all meshes have norm = 1
    "num_workers": 0,             # num worker for dataloader
    "save_epochs":[0,0 ,0,0,0,0,0,0,0,0,0,0],
}





## folder where .pt-files will be stored 
config["root_dir"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean\pt"
## folder containing input ply-files
config["folder"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean"
##if add_data is True
config["folder_add"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\Augmentation\Samples\linear sqrt(n)'
#for sym transform
config["mat_path"] = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"
## folder containing eigenvalues and eigenvectors
config["path_decomp"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'


i=0
result_df = pd.DataFrame(columns=["nb_freq","FFW","use_act","act","sym","p","epoch","best_epoch","MAE_VAL","EUC_VAL","MAE_TRAIN","EUC_TRAIN","MIN_EUC_TRAIN"]) 
if __name__ == "__main__":
    for ffw in [False,True]:
        for u in [0,1,2,3,-1]:
            if u == 0:
                act_list = ["ELU"]
            else:
                act_list = ["ELU","ReLU","Tanh","Sigmoid"]
            for act in act_list:
                for s in [False,"sym","asym"]:
                    for p in [0,0.3]:
                        np.random.seed(31415)
                        torch.manual_seed(31415)
                        random.seed(31415)
                        os.environ['PYTHONHASHSEED'] = str(31415)
               
                        config["save_epoch"] = 0
                        config["FFW"] = ffw
                        config["use_activation"] = u
                        config["activation"] = act
                        config["sym"] = s
                        config["p"] = p
                        
                        
                        
                        results = train_bf(config)
                        settings = [ffw,u,act,s,p]
                        result_df.loc[i]= [results[0]] + settings + results[1:]
                        i = i+1
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                            print(result_df)
                        
    
    print(config)