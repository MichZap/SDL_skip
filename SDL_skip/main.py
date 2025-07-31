import os
import torch
import numpy as np
import random
import pandas as pd
from train import train_bf




torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(True)

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
    "shed_pat":25,               # learning rate sheduler patience, see N in paper
    "es_pat":100,                 # early stopping patience, see M in paper
    "min_delta":0.0000005,                # parameter for early stopping tolerance
    
    #model base settings:
    "depth": 3,                   # model depth 0 pools directly to the latent size 
    "nb_freq": [8192,4096,2048,1024,512,256,128],                # number of eigenvectors considered, here i,m and j from illustration
    "hidden_dim": [64]*16,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
    "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
    "use_eigenvals":True,         # use eigenvalue encoding
    "conv_size": [1024]*16,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
    "prior_coef": 0.4,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
    
    #model additional settings:
    "bias":False,                 # use bias for linear layers
    "dropout":0,
    "reset":True,                # use axvier uniform weight initialization instead of pytorch default
    "aw":True,                    # use additional weight to scale every SkipBlock
    "wdim":False,                  # advanced aggregation if True one weight by vertex
    "Skip":2,                     # 0 no skip connections, 1 skip connection to end of each block, 2 skip connection to middle and end, 3 skip connection only to middle
    "learn":True,                 # False disables the learnable component of all SkipBlocks apart of the deepest part of the skipping pyramid, reduces number of parameters approx by a factor of 3
    "direct":False,                # if True the second skip conncetion start from the input
    "FFW":False,                   # additonal FFW layer to denosie the latent space for spectral decomp   
    
    #activation settings:
    "use_activation":0,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
    "activation": "ReLU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU   
        
    #attention settings:
    "use_att":False,              # use Attention-Matrix for pooling
    "n_heads":0,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
    "use_norm":False,             # use LayerNorm before every block of linear layers
    "init_a":0,                   # replaces LayerNorm by dynamic Tanh, use_norm = True and init_a != 0 
    "use_norm_att":False,        # in case of use_att True determine if the attentionscore gets normalized
    "sqrt":False,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
    "value":False,                # in case of multihead attention, use additional 3x3 value matrix
    "vinit":False,                 # vertex initialization for attention matrix
    "eb_dim":0,                  # in case of vertex initialization: None => no embedding, else verteices are embedded to eb_dim before attention

    #augmentation settings
    "p":0.5,                      # augmentation probability for flipping and rotation per axis
    "k":1,                        # max amount of consecutive augmentations
    "deg":180,                     #rotation degree for augmentation
    "flip":False,                 #False disables flipping augmentation
    "rot":False,                 #Rotation augmentation
    "sigma":0,                   #Sigma param for guassian noise augmentation sigma = 0 => disabled
    "comp":100,                    #number of components used for spectral interpolation augmentation, comp=0 => disabled
    "aug_mean":False,            #False is default, True augmentations are done to the original instead of the deviations to the mean
    "add_data":False,             # if we use addition data from folder_add, not considered in train test split
    "sym":"sym",                  # False default, "sym" adds mirrored and symmetric mesh, "aysm" additional transfers asymtries from one face to another requires indices for mirroring
    "p_asym":0,                 # probability for asymmetrty transfer to control ratio between real and asymmetric data

    #SDL_simple                    Similar to SDL but no with constant width and advanced aggregation by spectral dim
    "simple":False,                # uses simple model
    "gate":False,                  # for simple version gate the weight parameter if true
    "flatten":False,               # flatten of the hidden dim during pooling only work if use_att false and simpel True
    "proj":False,                   # if False uses weights instead of proj matrix in mh attention
    "Pai": False,                   # Pai conv in the end
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
    "unique":False,               # train test split by unique id
    "num_workers": 0,             # num worker for dataloader
    "save_epochs":[0,0,0,0,0,0,0,0,0,0],
}





## folder where .pt-files will be stored 
config["root_dir"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean\pt"
#config["root_dir"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\SDL'
## folder containing input ply-files
config["folder"] = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean"
#config["folder"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
##if add_data is True
config["folder_add"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\Augmentation\Samples\linear sqrt(n)'
#for sym transform
config["mat_path"] = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"
## folder containing eigenvalues and eigenvectors
config["path_decomp"] = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'


i=0
config["print"] = True
result_df = pd.DataFrame(columns=["nb_freq","epoch","best_epoch","MAE_VAL","EUC_VAL","MAE_TRAIN","EUC_TRAIN","MIN_EUC_TRAIN"]) 
if __name__ == "__main__":
    #3,5,11,21,42,85,170
    for f in [3,5,11,21,42,85,170,341,682,1365]:
            np.random.seed(31415)
            torch.manual_seed(31415)
            random.seed(31415)
            os.environ['PYTHONHASHSEED'] = str(31415)
            if config["depth"] == 0:
                config["nb_freq"] = [f]
            else:
                config["nb_freq"].append(f)
            config["save_epoch"] = config["save_epochs"][i]
            results = train_bf(config)
            result_df.loc[i]=results
            i = i+1
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                print(result_df)
            config["nb_freq"] = config["nb_freq"][:-1]
            config["print"] = False
    
    print(config)