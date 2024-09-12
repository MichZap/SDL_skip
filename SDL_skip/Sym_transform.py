# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:24:48 2024

@author: Michael
"""

import numpy as np
import os
import scipy.io
from plyfile import PlyData
from scipy.linalg import orthogonal_procrustes


#mat_path = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"
#ply_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new\008_16092014_procr.ply"
#ply_path_temp = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new\517_19082010_procr.ply"


def symtrans(ply_path,mat_path):
    #new path to save modified files
    folder, filename_with_extension = os.path.split(ply_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    new_path = folder + "\\sym\\" + filename_without_extension + "_sym_trans.ply"
    new_path2 = folder + "\\sym\\" + filename_without_extension + "_sym.ply"
    
    if os.path.isfile(new_path) or "sym" in filename_without_extension:
        return [new_path,new_path2], [filename_without_extension + "_sym_trans",filename_without_extension + "_sym"]
    
    else:
        #load 2d projection
        comm2D_verts = scipy.io.loadmat(mat_path)['comm2D_f'][0]["verts"][0]
        NV = comm2D_verts.shape[1]
        
        #calcualte symmetric indices
        symmIdxs = np.zeros(NV, dtype=int)
        
        for jv in range(NV):
            distances = (comm2D_verts[0, :] + comm2D_verts[0, jv])**2 + (comm2D_verts[1, :] - comm2D_verts[1, jv])**2
            #min_d = np.min(distances)
            symmIdxs[jv] = np.argmin(distances)
        
    
        #load ply    
        verts_init, ply =load_ply(ply_path)
        verts_cent = verts_init-np.mean(verts_init,0)
    
        #apply symmetry transform
        verts_sym = verts_init[symmIdxs]
        A = np.array([-verts_sym[:,0],verts_sym[:,1],verts_sym[:,2]]).T
        
        #apply procrustes transform
        R,_=orthogonal_procrustes(A,verts_cent)
        pos=np.matmul(A,R).astype('float32')
        
        export_ply(ply,new_path,pos)  
    
        #average both meshes for symmetic version
        avg = (pos + verts_cent)/2
        
        export_ply(ply,new_path2,avg)
        
    
        return [new_path,new_path2], [filename_without_extension + "_sym_trans",filename_without_extension + "_sym"]

def aysmtrans(ply_path_temp,ply_path,mat_path):
    
    #create output paths
    folder, filename_with_extension = os.path.split(ply_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    
    _, temp_name = os.path.split(ply_path_temp)
    temp_name, _ = os.path.splitext(temp_name )
    
    name = filename_without_extension + "_" + temp_name
    name_inv = temp_name + "_" + filename_without_extension
    
    name1 = name  + "_asym1"
    name2 = name  + "_asym2"
    name3 = name_inv  + "_asym1"
    name4 = name_inv  + "_asym2"
    
    new_path1 = folder + "\\sym\\" + name1 + ".ply"
    new_path2 = folder + "\\sym\\" + name2 + ".ply"
    new_path3 = folder + "\\sym\\" + name3 + ".ply"
    new_path4 = folder + "\\sym\\" + name4 + ".ply"
    
    if os.path.isfile(new_path1):
        return [new_path1,new_path2], [name1,name2]
    elif "sym" in filename_without_extension:
        print("hace nada")
    elif os.path.isfile(new_path3):
        return [new_path3,new_path4], [name3,name4]
    else:
    
    
        #make sure sym transform is applied and get names and paths
        Paths_temp, Names_temp = symtrans(ply_path_temp,mat_path)
        Paths, Names = symtrans(ply_path,mat_path)
        
        #load plys
        v_sym, ply_sym = load_ply(Paths[1]) #symertic original
        v_temp_sym, ply_temp_sym = load_ply(Paths_temp[1]) #symetric template
        v_temp_asym1, ply_temp_asym1 = load_ply(ply_path_temp) #asymetric template
        v_temp_asym2, ply_temp_asym2 = load_ply(Paths_temp[0]) #asymetric template but flipped
        
        #calculate norm to normalize different meshsizes
        norm_v_sym = norm(v_sym)
        norm_v_ts = norm(v_temp_sym)

        
        #calculate template differences
        rdiff1 = v_temp_asym1 - v_temp_sym
        rdiff2 = v_temp_asym2 - v_temp_sym
        #apply asym to the ply and scale
        v_asym1 = rdiff1*(norm_v_sym/norm_v_ts) + v_sym
        v_asym2 = rdiff2*(norm_v_sym/norm_v_ts) + v_sym
        
        export_ply(ply_sym,new_path1,v_asym1)
        export_ply(ply_sym,new_path2,v_asym2)
        
        return [new_path1,new_path2], [name1,name2]


def norm(array):
    
    return np.sqrt(((array - array.mean(0))**2).sum())
    
def load_ply(path):
    
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
    
    return verts_init, ply

def export_ply(ply,path,array):
    
    ply["vertex"]["x"] = array[:,0]
    ply["vertex"]["y"] = array[:,1]
    ply["vertex"]["z"] = array[:,2]
    
    modified_ply = PlyData(ply, text = False)
    modified_ply.write(path)

    
#print(symtrans(ply_path,mat_path))
#print(aysmtrans(ply_path_temp,ply_path,mat_path))    