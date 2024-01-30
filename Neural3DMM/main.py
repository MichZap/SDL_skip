import os
import torch
import numpy as np
import random
import pandas as pd
from train import train_bf


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

config = {

    "downsample_att": [True,True,True,True],
    "upsample_att": [True,True,True,True],
    "reference_points": [[3000]], 
    "dim_att": 21,
    "top_k_enc": 2,
    "top_k_dec": 32,
    "prior_init": True,
    "prior_coef": 0.8,
    "device": "cuda",
    "learning_rate": 1e-4,
    "train_batch_size": 64,
    "test_batch_size": 64,
    "activation":'elu',
    "prior_coef": 0.8,
    "step_sizes": [1, 1, 1, 1, 1],
    "dilation": None,
    "filters_enc": [[3, 16, 16, 16, 32],[[],[],[],[],[]]],
    "filters_dec": [[32, 16, 16, 16, 16],[[],[],[],[],3]],
    "p":0.3,
    "epochs":10000,
    "typ": "Coma",
    "shed_fac":0.75,
    "print_epoch":25,
    "shed_pat":25,
    "es_pat":100
}




config["root_dir"] = '//home/mzappe/Deep3DMM/Data/procrustes/Coma'
config["folder"] = '//home/mzappe/Deep3DMM/Data/Coma'
config["template_file_path"] = '//home/mzappe/Deep3DMM/Data/Coma/bareteeth.000001.ply'

i=0
result_df = pd.DataFrame(columns=["latent_size","epoch","best_epoch","MAE_VAL","EUC_VAL","MAE_TRAIN","EUC_TRAIN"]) 
for f in [8,16,32,64,128,256,512,1024,2048,4096]:
    np.random.seed(31415)
    torch.manual_seed(31415)
    random.seed(31415)
    os.environ['PYTHONHASHSEED'] = str(31415)
    
    config["size_latent"] = f
    results = train_bf(config)
    result_df.loc[i]=results
    i = i+1
    
    print(result_df)
result_df.to_csv("result_df_coma.csv")
