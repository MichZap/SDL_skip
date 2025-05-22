import os
import torch
import numpy as np
import random
import pandas as pd
from train import train_bf




torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


config = {
  "device": "cuda",
  "nb_freq": 896,
  "learning_rate": 1e-4,
  "train_batch_size": 16,
  "test_batch_size": 21,
  "size_conv": 3,
  "encoder_features": [3, 32, 64, 64, 64],
  "downsample":[512, 376, 256, 128],
  "dropout":0,
  "activation_func": "ELU",
  "p":0.7,
  "flip":False,
  "rot":False,
  "sigma":0,
  "comp":100,
  "aug_mean":False,
  "epochs":10000,
  "typ": "BBF",
  "shed_fac":0.5,
  "print_epoch":25,
  "shed_pat":200,
  "es_pat":500
}





root_dir=r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\mean'+'//'+str(config["nb_freq"])
config["root_dir"] = root_dir
folder=r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
config["folder"] = folder
path_decomp = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'
config["path_decomp"] = path_decomp


i=0
result_df = pd.DataFrame(columns=["latent_size","epoch","best_epoch","MAE_VAL","EUC_VAL","MAE_TRAIN","EUC_TRAIN"]) 
for f in [9,15,33,63,126,255,510,1023,2046,4095]:
    
    np.random.seed(31415)
    torch.manual_seed(31415)
    random.seed(31415)
    os.environ['PYTHONHASHSEED'] = str(31415)
    
    config["size_latent"] = f
    results = train_bf(config)
    result_df.loc[i]=results
    i = i+1
    print(result_df)

