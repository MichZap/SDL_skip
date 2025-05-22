# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:19:30 2024

@author: Michael
"""

from eval_adddata import create_df
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
#folder = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\PCA\aug\255\True"
#5.6530735e-05
folder = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\SDL_skip\paper\85"
#6.5961776

df = create_df(folder)

# Function to calculate the per-vertex distance between two meshes
def calculate_per_vertex_distance(mesh1, mesh2):
    # Calculate Euclidean distance for each corresponding vertex
    l2 = (mesh1 -mesh2)**2
    return np.average(np.sqrt(np.sum(l2,axis=1)))

def read_path(path):
        ply = PlyData.read(path)
        vertex=ply["vertex"]
        verts_init=np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)  
        return verts_init

# Function to calculate the diversity score
def calculate_diversity(data):
    pair_distances = []
    Paths = data.Path.tolist()
    Paths2 = Paths.copy()
    # Compute the distance between every unique pair of meshes
    for path1 in tqdm(Paths, desc="Calculating Diversity"):
        Paths2.remove(path1)
        mesh1 = read_path(path1)
        
        for path2 in tqdm(Paths2, desc="Calculating Diversity again"):
            mesh2 =read_path(path2)
            pair_distances.append(calculate_per_vertex_distance(mesh1, mesh2))
            print(np.mean(pair_distances))
            
    # Return the average pairwise distance
    return np.mean(pair_distances)

calculate_diversity(df)