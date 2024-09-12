# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 00:32:16 2024

@author: Michael
"""
import torch
import os
import glob
from plyfile import PlyData
import numpy as np
import pandas as pd
from utilis import meanply
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

folder = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
outpath =  r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\Augmentation\Samples"
n_samples =100


def exportply(array,path,ply):
    ply["vertex"]["x"] = array[:,0]
    ply["vertex"]["y"] = array[:,1]
    ply["vertex"]["z"] = array[:,2]
    now = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
    new_path = path+"/"+"linear"+"_"+str(now)+".ply"
    ply=PlyData(ply, text = False)
    ply.write(new_path)

#create df
Name=[]
os.chdir(folder)
for file in glob.glob("*.ply"):
    Name.append(file.partition(".")[0])

Path=[folder +"\\"+name+".ply" for name in Name]

Babyface_df=pd.DataFrame({'Name' : Name,'Path' : Path})

np.random.seed(31415)
torch.manual_seed(31415)

df_train,df_val = train_test_split(Babyface_df,test_size = 0.2)

mean=meanply(df_train, typ = "BBF")

n = len(df_train)
data = []
for idx in range(n):
    raw_path = df_train.Path.iloc[idx]
    ply = PlyData.read(raw_path)
    vertex=ply["vertex"]
    verts_init=np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)-mean
    data.append(verts_init)
    

for i in range(n_samples):
    time.sleep(1)
    alphas = np.random.normal(0,1/np.sqrt(n),n)
    #alphas = alphas/np.sum(alphas)
    sample = verts_init*0
    for i in range(n):
        sample += alphas[i]*data[i]+mean/n
    
    exportply(sample,outpath,ply)
