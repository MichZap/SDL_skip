import torch
import numpy as np
from utilis import Mesh
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.linalg import eigh

def spec_dev(in_path,out_path):
    
    #caluclate eigenvalues and eigenvectors of the graph laplacian
    #input mesh consists of vertices and faces
    
    #import template mesh
    template_mesh = Mesh(filename=in_path)
    #tranform to torch tensors
    faces = [torch.tensor(fa, dtype=torch.long) for fa in template_mesh.f]
    face = torch.stack(faces, dim=-1)
    
    #transform to pytorch geometric Data and transform faces to edges
    data_mesh = Data(x = torch.tensor(template_mesh.v), face=face)
    transform = FaceToEdge()
    data_mesh = transform(data_mesh)
    
    #calculate graph laplacian
    GL = get_laplacian(data_mesh.edge_index)
    L = to_scipy_sparse_matrix(GL[0], GL[1], data_mesh.num_nodes)
    
    #compute eiegenvalues and eigenvectors
    eig_vals, eig_vecs = eigh(L.todense())
    
    #store eigenvalues and vectors
    outfile_vec = out_path +"\\"+ str(data_mesh.num_nodes) +"eig_vecs.npy"
    outfile_val = out_path +"\\"+ str(data_mesh.num_nodes) +"eig_vals.npy"
    
    np.save(outfile_vec, eig_vecs)
    np.save(outfile_val, eig_vals)