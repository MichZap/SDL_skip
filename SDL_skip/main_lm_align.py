# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:01:12 2025

@author: Michael
"""
import numpy as np
import pandas as pd
import os
import glob
import scipy.io as sio

from plyfile import PlyData
from vertextosurface import load_meshlab_pp_xml, align_meshes_by_landmarks, export_ply, compute_vertex_to_surface_distance

#input path:
path_new_cases = r"D:\WhasingtonData\OriginalScans"
path_df = r"C:\Users\Michael\Downloads\3DMeshes\T9a3\bbdf.xlsx"
path_pts =r"D:\WhasingtonData\OriginalScans\landmarks23" #only for normdf

#path_matlab for template landmarks
path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\FaceModel_GMM_MVN.mat"
mat_path = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat" #for RFL Meshes

#output path:
path_out = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\SDL_skip\normdf'


#load df
dff = pd.read_excel(path_df,"normaldf")
dff = dff[["Subfolder","File"]]

# for each subfolder, look for exactly one .obj and one .pp
obj_paths = []
pp_paths  = []

for sub in dff['Subfolder']:
    if sub != "asd":
        base_dir = os.path.join(path_new_cases, sub)
        #obj = glob.glob(os.path.join(base_dir, '*.obj'))[0]
        #pp  = glob.glob(os.path.join(base_dir, '*.pp'))[0]
        
        # Find STL file(s)
        stl_files = glob.glob(os.path.join(base_dir, '*.stl'))
        if not stl_files:
            print(f"Skipping {sub}: no STL file found")
            obj_paths.append("NA")
            pp_paths.append("NA")
            continue  # Skip to next sub
        
        obj = stl_files[0]
        s = sub[5:]+".pts"
        pts = os.path.join(path_pts,s)
    
        obj_paths.append(obj)
        #pp_paths.append(pp)
        pp_paths.append(pts)
    else:
        obj_paths.append("NA")
        pp_paths.append("NA")

dff['obj_path'] = obj_paths
dff['pp_path']  = pp_paths
dff = dff[dff['obj_path']!="NA"]

###read and process template landmarks
landmark_verts = list(sio.loadmat(path)['FaceModel']['landmark_verts'][0][0][0]-1)#matlab indexing starts at 1
names = sio.loadmat(path)['FaceModel']['landmark_names']

n = len(names[0][0][0])

landmark_names = []

for i in range(n):
    landmark_names.append(names[0][0][0][i][0])
    

for idx in dff.index:
    path_obj = dff['obj_path'][idx]
    path_pp = dff['pp_path'][idx]
    path_ply = dff['File'][idx]
    name = dff["Subfolder"][idx]+".ply"
    print(dff['Subfolder'][idx],":")
    
    if pd.notna(path_ply):
        ###load template  
        ply = PlyData.read(path_ply)
        vertex = ply["vertex"]
        verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
        
        #undo reflections
        if "RFL" in path_ply: 
            #load 2d projection
            comm2D_verts = sio.loadmat(mat_path)['comm2D_f'][0]["verts"][0]
            NV = comm2D_verts.shape[1]
            
            #calcualte symmetric indices
            symmIdxs = np.zeros(NV, dtype=int)
            
            for jv in range(NV):
                distances = (comm2D_verts[0, :] + comm2D_verts[0, jv])**2 + (comm2D_verts[1, :] - comm2D_verts[1, jv])**2
                #min_d = np.min(distances)
                symmIdxs[jv] = np.argmin(distances)
            
            verts_init = verts_init[symmIdxs]
            verts_init[:,0] = -verts_init[:,0]
        
        #filter landmarks
        verts = verts_init[landmark_verts]  
        #create df
        df_temp = pd.DataFrame(verts, columns = ["x","y","z"])
        df_temp["names"] = landmark_names
        #load original landmarks
        landmark_verts_org, landmark_names_org = load_meshlab_pp_xml(path_pp)
    
        df_org = pd.DataFrame(landmark_verts_org, columns = ["x","y","z"])
        df_org["names"] = landmark_names_org
        #combine landmark dfs org and template, sucht that they have the same order
        df = pd.merge(df_temp,df_org, on = "names" , how = "inner")
        #remove rows with empty landmarks
        df = df[~((df['x_y'] == 0) & (df['y_y'] == 0) & (df['z_y'] == 0))]
        df = df[~(df[['x_y', 'y_y', 'z_y']].isna().all(axis=1))]
        
        #align template mesh to original by landmarks:
        out = align_meshes_by_landmarks(verts_init, df)
        #compute vertex to surface distance by vertex of the template mesh
        distances,_ = compute_vertex_to_surface_distance(out,path_obj,landmark_verts_org)
        #export colored and aligned template mesh
        outpath = os.path.join(path_out, name)
        if distances[0] !=0:
            export_ply(ply,outpath,out,distances,True,0,5)
    
    
