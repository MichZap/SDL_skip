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
  "depth": 2,
  "nb_freq": [4096,1024,256],
  "device": "cuda",
  "learning_rate": 1e-3/2,
  "train_batch_size": 16,
  "test_batch_size": 16,
  "hidden_dim": [64]*7,
  "conv_size": [1024]*7,
  "prior_coef": 0.4,
  "use_activation":0,
  "dropout":0,
  "activation": "ELU",
  "use_conv": True,
  "use_eigenvals":True,
  "use_norm":False,
  "use_att":False,
  "n_heads":1,
  "bias":False,
  "p":0,
  "reset":True,
  "aw":True,
  "epochs":10000,
  "typ": "BBF",
  "shed_fac":0.75,
  "print_epoch":25,
  "shed_pat":25,
  "es_pat":100,
  "Skip":2,
  "value":True,
  "learn":False,
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

