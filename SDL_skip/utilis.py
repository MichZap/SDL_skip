import torch
from plyfile import PlyData
import numpy as np
import torch.nn.functional as F
import os
from scipy.linalg import orthogonal_procrustes
from itertools import combinations

#from celi/antonias code
def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
    

          
#calculate mean face
def meanply(Data, typ = "Coma", sym = False, norm = False):
    
    if sym != False:
        sym = True
    
    file = f"mean_sym_{int(sym)}_SSH_{int(norm)}_{typ}.npy"
        
    if os.path.isfile(file):
        m = np.load(file)
    else:
        i=0
        if typ == "FWH":
            n = int(Data.shape[1]/3)
            for idx in range(len(Data)):
                vertex = Data[idx]
                verts_init = np.stack((vertex[:n], vertex[n:2*n], vertex[2*n:3*n]),axis=1)
                verts_init = verts_init.astype('float32')
                SSH = np.sqrt(((verts_init - verts_init.mean(0))**2).sum()) if norm == True else 1
                verts_init = verts_init/SSH
                
                if i==0:
                    mean=verts_init
                    i=i+1
                else:
                    mean=mean+verts_init
        else:
            for idx in Data.index:
                raw_path = Data.Path[idx]
                ply = PlyData.read(raw_path)
                vertex=ply["vertex"]
                verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
                verts_init = verts_init.astype('float32')
                SSH = np.sqrt(((verts_init - verts_init.mean(0))**2).sum())/verts_init.shape[0] if norm == True else 1
                verts_init = verts_init/SSH
                
    
                if i==0:
                    mean=verts_init
                    i=i+1
                else:
                    mean=mean+verts_init
                    
        
        sc = 1000 if typ == "Coma" else 1    
        m = mean/len(Data)*sc
        np.save(file,m)
    return(m)

def sdply(Data, typ = "Coma", sym = False):
    
    if sym == "asym":
        sym = "sym"
    
    file = "sd_sym"+typ+".npy" if sym == "sym" else "sd_"+typ+".npy"
    
    if os.path.isfile(file):
        sd = np.load(file)
        
    else:
        i=0
        m = meanply(Data, typ)
        
        if typ == "FWH":
            n = int(Data.shape[1]/3)
            for idx in range(len(Data)):
                vertex = Data[idx]
                verts_init = np.stack((vertex[:n], vertex[n:2*n], vertex[2*n:3*n]),axis=1)
                verts_init = verts_init.astype('float32')-m
    
                if i==0:
                    sd = verts_init**2
                    i=i+1
                else:
                    sd = sd + verts_init**2
        else:
            for idx in Data.index:
                
                sc = 1000 if typ == "Coma" else 1  
                
                raw_path = Data.Path[idx]
                ply = PlyData.read(raw_path)
                vertex=ply["vertex"]
                verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
                verts_init = verts_init.astype('float32')*sc - m
    
                if i==0:
                    sd = verts_init**2
                    i=i+1
                else:
                    sd = sd + verts_init**2
        
  
        sd = np.sqrt(sd/(len(Data)-1))
        np.save(file,sd)
    return(sd)
    
    

def procrustes_ply(Data,folder):
    
    i=0
    for idx in range(len(Data)):
        
        raw_path = Data.Path[idx]
        Name_org = Data.Name[idx]
        Name_new = Name_org + "_procr"
        
        
        #new_path = folder + "\\" + Name_new +".ply"
        new_path = os.path.join(folder, Name_new + ".ply")
        print(new_path)
        
        if os.path.isfile(new_path) == False:
            
            ply = PlyData.read(raw_path)

            vertex=ply["vertex"]
            verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
            verts_init = verts_init.astype('float32')

            verts_cent = verts_init-np.mean(verts_init,0)


            if i==0:
               pos_base = verts_cent
               
               i=1
            A=verts_cent

            R,_=orthogonal_procrustes(A,pos_base)
            pos=np.matmul(A,R).astype('float32')

            ply["vertex"]['x'] = pos[:,0]
            ply["vertex"]['y'] = pos[:,1]
            ply["vertex"]['z'] = pos[:,2]

            ply=PlyData(ply, text = False)

            ply.write(new_path)

class EUCLoss(torch.nn.Module):
    def __init__(self):
        super(EUCLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction="none")
        loss = torch.mean(torch.sqrt(torch.sum(loss,axis =2)))
        return loss
    
class Mesh(object): 
    def __init__(self,
                 v = None,
                 f = None,
                 Adj = None,
                 filename = None,
                 ):
        
        if filename == None:

            self.v = v
            self.f = f
            self.Adj = self.get_adjacency(f,v)
        else:

            self.v,self.f = self.read(filename)
            self.Adj = self.get_adjacency(self.f,self.v)
        
        
    def read(self,filename):
        
        ply = PlyData.read(filename)
        
        vertex = ply["vertex"]
        face = ply["face"]['vertex_indices']
        
        v=np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
        f=np.stack(face,axis=0)
        
        return v,f
    
    def get_adjacency(self,f,v):
        
        dim = v.shape[0]
        
        Adj = np.eye(dim)
        
        for face in f:
            # Get all 2-dimensional combinations
            combinations_2d = list(combinations(face, 2))
            for comb in combinations_2d:
                Adj[comb] = 1
                Adj[comb[::-1]] = 1
                
        return Adj
            
def identify_subject(name):
    if name.startswith('uu5t1_LSCM'):
        return name[:17]
    elif name.startswith('CNMC'):
        return name[:9]
    elif name[0].isdigit():
        return name[:3]
    else:
        return None  # Or some default value if the conditions aren't met


    
