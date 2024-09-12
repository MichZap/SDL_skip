# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:06:14 2024

@author: Michael
"""
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from utilis import meanply
from plyfile import PlyData
from datetime import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt

path_decomp = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'
folder = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
outpath =  r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\Augmentation\Samples"
typ = "BBF"
freq = 200
n_samples =100
bootstrap = False

f = 31027 
 
#load eigenvectors and value
outfile_vec = path_decomp +"\\"+ str(f) +"eig_vecs.npy"
outfile_val = path_decomp +"\\"+ str(f) +"eig_vals.npy"

eig_vecs = np.load(outfile_vec)
eig_vecs = eig_vecs[:,:freq]

#create df
Name=[]

os.chdir(folder)
for file in glob.glob("*.ply"):
    Name.append(file.partition(".")[0])

Path=[folder +"\\"+name+".ply" for name in Name]

Babyface_df=pd.DataFrame({'Name' : Name,'Path' : Path})

np.random.seed(31415)


df_train,df_val = train_test_split(Babyface_df,test_size = 0.2)
                         
mean=meanply(df_train, typ = typ)

k = len(df_train)
lat_list=[]
for idx in range(k):
    raw_path = df_train.Path.iloc[idx]
    ply = PlyData.read(raw_path)
    vertex = ply["vertex"]
    verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)-mean
    lat = np.matmul(eig_vecs.T,verts_init)
    lat_list.append(lat)




def create_mapped_array(randarray, list_of_arrays):
    mapped_array = np.empty_like(randarray, dtype=float)
    for i in range(randarray.shape[0]):
        for j in range(randarray.shape[1]):
            mapped_array[i,j] = list_of_arrays[randarray[i,j]][i,j]
    return mapped_array 

def exportply(array,path,ply,freq):
    ply["vertex"]["x"] = array[:,0]
    ply["vertex"]["y"] = array[:,1]
    ply["vertex"]["z"] = array[:,2]
    now = datetime.now().strftime("%Y-%m-%d%H-%M-%S")
    new_path = path+"/"+str(freq)+"_"+str(now)+".ply"
    ply=PlyData(ply, text = False)
    ply.write(new_path)

#sample
if bootstrap:
    for i in range(n_samples):
        time.sleep(1)
        randarray = np.random.randint(0, k, (freq, 3))
        mapped_array = create_mapped_array(randarray, lat_list)
        sample = np.matmul(eig_vecs,mapped_array)+mean
        exportply(sample,outpath,ply,freq)


else:
    lat_list_array = np.array(lat_list)
    reshaped_array = lat_list_array.reshape((92,-1))


    corr = np.corrcoef(reshaped_array, rowvar=False)
    mean_l = np.mean(reshaped_array,axis = 0)
    sd = np.std(reshaped_array,axis = 0)
    diag_sd = np.diag(sd)
    cov = diag_sd @ corr @ diag_sd

    #sns.lineplot(mean_l)
    #sns.lineplot(sd)

    #plt.figure(figsize=(10, 10))
    #ax = sns.heatmap(corr,cmap='coolwarm', center=0)

    rvec = np.random.multivariate_normal(mean_l,cov,n_samples)
    #sns.lineplot(rvec.mean(axis=0))
    rvec_r = rvec.reshape(100,freq,3)
    samples = np.matmul(eig_vecs,rvec_r) + mean
    for i in range(n_samples):
        time.sleep(1)
        exportply(samples[i],outpath,ply,freq)

