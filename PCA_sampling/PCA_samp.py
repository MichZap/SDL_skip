# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:41:50 2024

@author: Michael
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from plyfile import PlyData

k = 255
aug = True
ind = True
nsamples = 1000

path_noaug = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\cluster\PCA_BBF\spectral latent size\principal_components_20241112_170026.npy"
path_aug = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\cluster\PCA_BBF\spectral latent size\principal_components_20241112_170709.npy"
folder = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes'
m_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\mean_BBF.npy"
out_path = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\PCA'
example_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\Data\Synthetic_3D_face_00001.ply"

train_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\cluster\PCA_BBF\train_data.npy"

mean = np.load(m_path)
if aug == True:
    path = path_aug
    out_path = out_path+"//aug//"+str(k)+"//"+str(ind)
else:
    path = path_noaug+str(k)+"//"+str(ind)

def create_df(folder):
    Name=[]
    
    os.chdir(folder)
    for file in glob.glob("*.ply"):
        Name.append(file.partition(".")[0])
    
    Path = [folder +"\\"+name+".ply" for name in Name]
    Babyface_df = pd.DataFrame({'Name' : Name,'Path' : Path})
    
    
    Babyface_df = pd.DataFrame({'Name' : Name,'Path' : Path})
    return Babyface_df

def export_ply(ply,path,array):
    
    ply["vertex"]["x"] = array[:,0]
    ply["vertex"]["y"] = array[:,1]
    ply["vertex"]["z"] = array[:,2]
    
    modified_ply = PlyData(ply, text = False)
    modified_ply.write(path)

def Latent_space(path, mean, folder,k,plot = False):
    Babyface_df_test = create_df(folder)
    
    Lat_space = []
    pca_m = np.load(path)[:k,:]
    for path in Babyface_df_test.Path:
        ply = PlyData.read(path)
        vertex = ply["vertex"]
        verts = np.append(vertex['x'], [vertex['y'], vertex['z']]) - np.append(mean[:,0],[mean[:,1],mean[:,2]])
        lat = np.matmul(pca_m,verts)
        if plot:
           plt.hist(lat, bins=30, alpha=0.7, color="#ff7f0e", edgecolor="black")
           plt.show() 
        Lat_space.append(lat)
    
    return Lat_space, Babyface_df_test

Lat_space, Babyface_df_test = Latent_space(path, mean, folder, k =k, plot = False)

corr = np.corrcoef(Lat_space, rowvar=False)
plt.imshow(corr, cmap="PiYG", interpolation='nearest')
plt.show()


def sample(n_samples,path,k,ply,mean,array,out_path,ind):
       
    if ind:
        sd = np.std(array,axis = 0)
        cov = np.diag(sd)
    else:
        cov = np.cov(array, rowvar=False)
    
    m = np.mean(array, axis = 0)
    pca_m = np.load(path)[:k,:]    
        
    for name in range(n_samples):
        vec = np.random.multivariate_normal(m,cov)

        out = np.matmul(pca_m.T,vec) + np.append(mean[:,0],[mean[:,1],mean[:,2]])
        out = out.reshape(3,-1).T
        SSH = np.sqrt(((out - out.mean(0))**2).sum())
        out = out/SSH
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        path = out_path+"/"+str(name)+"_k="+str(k)+"_ind="+str(ind)+".ply"
        export_ply(ply,path,out)


    
ply = PlyData.read(example_path)
sample(nsamples,path,k,ply,mean,Lat_space,out_path,ind)

