# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:24:48 2024

@author: Michael
"""

import numpy as np
import os
import scipy.io as sio
from plyfile import PlyData

from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as RT


#mat_path = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"
#fm_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\FaceModel_GMM_MVN.mat"
#ply_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new\008_16092014_procr.ply"
#ply_path_temp = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new\517_19082010_procr.ply"


def symtrans(ply_path,mat_path,fm_path = None):
    #new path to save modified files
    folder, filename_with_extension = os.path.split(ply_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)

    #new_path = folder + "\\sym\\" + filename_without_extension + "_sym_trans.ply"
    #new_path2 = folder + "\\sym\\" + filename_without_extension + "_sym.ply"
    new_path =  os.path.join(folder, "sym", filename_without_extension + "_sym_trans.ply")
    new_path2 = os.path.join(folder, "sym", filename_without_extension + "_sym.ply")
    
    if os.path.isfile(new_path) or "sym" in filename_without_extension:
        return [new_path,new_path2], [filename_without_extension + "_sym_trans",filename_without_extension + "_sym"]
    
    else:
        #load 2d projection
        comm2D_verts = sio.loadmat(mat_path)['comm2D_f'][0]["verts"][0]
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
        
        if fm_path != None:
            #align mesh such that the center in between the eyes is 0,0,0 and the eyes lie on 1,0,0
            landmark_verts = list(sio.loadmat(fm_path)['FaceModel']['landmark_verts'][0][0][0]-1)
            exR_idx = landmark_verts[1]
            exL_idx = landmark_verts[3]
            prn_idx = landmark_verts[7]
            
            exR = verts_cent[exR_idx]
            exL = verts_cent[exL_idx]
            prn = verts_cent[prn_idx]
            
            verts_init = align_mesh(exR,exL,prn,verts_init)
    
        #apply symmetry transform
        verts_sym = verts_init[symmIdxs]
        A = np.array([-verts_sym[:,0],verts_sym[:,1],verts_sym[:,2]]).T
        
        if fm_path == None:
            #apply procrustes transform
            R,_ = orthogonal_procrustes(A,verts_cent)
            pos = np.matmul(A,R).astype('float32')
        else:
            pos = A
        
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
    
    #new_path1 = folder + "\\sym\\" + name1 + ".ply"
    #new_path2 = folder + "\\sym\\" + name2 + ".ply"
    #new_path3 = folder + "\\sym\\" + name3 + ".ply"
    #new_path4 = folder + "\\sym\\" + name4 + ".ply"
    
    new_path1 = os.path.join(folder, "sym", name1 + ".ply")
    new_path2 = os.path.join(folder, "sym", name2 + ".ply")
    new_path3 = os.path.join(folder, "sym", name3 + ".ply")
    new_path4 = os.path.join(folder, "sym", name4 + ".ply")
    
    
    
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

def align_vector(v1, v2):

    #gives matrix to rotate v1 to v2    

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_theta)

    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        return np.eye(3)  # No rotation needed
    else:
        r = RT.from_rotvec((axis / axis_norm) * angle)
        return r.as_matrix()

def align_mesh(exR,exL,prn,x):
    #aligns mesh by eyes and nose landmarks with x and y axis
    
    eye_vec = exR - exL                     # Vector from left to right eye
    eye_center = (exL + exR) / 2            # Midpoint between eyes
    T = -eye_center                         # Translate such that eye_center is at origin

    # Align eye vector to X-axis
    R1 = align_vector(eye_vec, np.array([1.0, 0.0, 0.0]))
    
    x = (x + T) @ R1
    exL = (exL + T) @ R1
    exR = (exR + T) @ R1
    prn = (prn + T) @ R1
    
    # Align face normal to Z-axis
    v_eye = exR - exL
    v_nose = prn - (exL + exR) / 2
    face_normal = np.cross(v_eye, v_nose)
    R2 = align_vector(face_normal, np.array([0.0, 0.0, 1.0]))
    x = x  @ R2

    return x

#calculate P
# from scipy.sparse import csr_matrix

# NV = len(symmIdxs)
# row_indices = np.arange(NV)
# col_indices = symmIdxs
# data = np.ones(NV)

# P = csr_matrix((data, (row_indices, col_indices)), shape=(NV, NV))
# sp.save_npz(r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\SymEqui\tmp\P.npz', P)
# Convert to dense numpy array
#P_dense = P.toarray()

# Convert to PyTorch tensor
#P_tensor = torch.from_numpy(P_dense).float()

# Compute eigenvalues and eigenvectors
#_, E = torch.linalg.eig(P_tensor)
#E = E.real
#torch.save(E,r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\SymEqui\tmp\E.pt')
    
#print(symtrans(ply_path,mat_path))
#print(aysmtrans(ply_path_temp,ply_path,mat_path))    