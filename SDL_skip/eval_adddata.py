# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:50:24 2024

@author: Michael
"""
#code to validate results on additional test set


import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from plyfile import PlyData, PlyElement
from utilis import procrustes_ply, EUCLoss
from Dataloader import autoencoder_dataset
from torch.utils.data import DataLoader
from SDL import SDL_skip
from Sym_transform import symtrans, aysmtrans

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Settings
#Settings
config = {
  "depth": 0,                   # model depth 0 pools directly to the latent size 
  "nb_freq": [],   # number of eigenvectors considered, here i,m and j from illustration
  "device": "cuda",
  "learning_rate": 1e-3/2,
  "train_batch_size": 1,
  "test_batch_size": 1,
  "hidden_dim": [64]*8,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "conv_size": [1024]*8,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "prior_coef": 0.8,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
  "use_activation":0,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
  "dropout":0.0,              
  "activation": "SiLU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU 
  "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
  "use_eigenvals":True,         # use eigenvalue encoding
  "use_norm":False,             # use LayerNorm before every block of linear layers
  "use_att":False,              # use Attention-Matrix for pooling
  "use_norm_att":"Soft",          # in case of use_att True determine if the attentionscore gets normalized
  "sqrt":True,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
  "n_heads":8,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
  "value":False,                 # in case of multihead attention, use additional 3x3 value matrix
  "bias":False,                 # use bias for linear layers
  "p":0,                        # augmentation probability for flipping and rotation per axis
  "k":2,                        # max amount of consecutive augmentations
  "reset":False,                 # use axvier uniform weight initialization instead of pytorch default
  "aw":True,                    # use additional weight to scale every SkipBlock
  "epochs":10000,
  "typ": "BBF",                 # data to be used "BBF", "Coma" and "FWH"
  "shed_fac":0.75,              # learning rate scale factor on plateau, see X in paper
  "print_epoch":25,
  "shed_pat":25,                # learning rate sheduler patience, see N in paper
  "es_pat":100,                 # early stopping patience, see M in paper
  "Skip":2,                     # 0 no skip connections, 1 skip connection to end of each block, 2 skip connection to middle and end, 3 skip connection only to middle
  "learn":True,                # False disables the learnable component of all SkipBlocks apart of the deepest part of the skipping pyramid, reduces number of parameters approx by a factor fo 3
  "simple":False,                # uses simple model
  "direct":True,                # if True the second skip conncetion start from the input
  "add_data":False,             # if we use addition data from folder_add, not considered in train test split
  "sym":False,                 # False default, "sym" adds mirrored and symmetric mesh, "aysm" additional transfers asymtries from one face to another requires indices for mirroring
  "p_asym":0.1,                 # probability for asymmetrty transfer to control ratio between real and asymmetric data
  "min_delta":0,                # parameter for early stopping tolerance
  "sd":False,                   # sd normalization in Dataloader
  "gate":False,                 # for simple version gate the weight parameter if true
  "flatten":False,              # flatten of the hidden dim during pooling only work if use_att false and simpel True
  "SSH": False,                  # normalizetion in data loader such that all meshes have norm = 1
  "proj":True,                  # if False uses weights instead of proj matrix in mh attention
  "Pai": True,                  # Pai conv in the end
  "paiconv_size":[9,9],             # Pai conv size
  "in_ch":[3,3],                    # Pai conv in channels
  "out_ch":[3,3],                   # Pai conv out channels
  "pai_act":[1,1],                # Pai conv activation  0 or 1
  "pai_skip":[True,True],
  "pai_small":[False,False],
  "save_epochs":[1177,0,0,0,0,0,0,0,0,0,0,0],
}

## folder where .pt-files will be stored 
root_dir = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\SDL'
## folder containing test ply-files
folder = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\addTestData'
## path to a procrustes alligned ply file from the training data
path_train = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\uu5t1_LSCM_fix573_190903120253_R_FNN_woHat_procr.ply"
## folder containing eigenvalues and eigenvectors
path_decomp = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'

out_path = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\SDL_skip'
example_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\Data\Synthetic_3D_face_00001.ply"

#max error in mm for color coding
m = 3

def create_df(folder,proc = False,sym = False, mat_path = None):
    Name=[]
    
    os.chdir(folder)
    for file in glob.glob("*.ply"):
        Name.append(file.partition(".")[0])
    
    Path=[folder +"\\"+name+".ply" for name in Name]
    Babyface_df = pd.DataFrame({'Name' : Name,'Path' : Path})
    
    if sym == "sym" or sym == "asym":
        for path in Babyface_df.Path:
            newpaths,newnames = symtrans(path,mat_path)

            Path.extend(newpaths)
            Name.extend(newnames)
        if sym == "asym":
            for path in Babyface_df.Path:
                for path1 in Babyface_df.Path:
                    if path != path1:
                        newpaths,newnames = aysmtrans(path,path1,mat_path)
                        Path.extend(newpaths)
                        Name.extend(newnames)
        
    if proc:
        Name.insert(0,"uu5t1_LSCM_fix573_190903120253_R_FNN_woHat_procr")
        Path.insert(0,path_train)
    
    Babyface_df = pd.DataFrame({'Name' : Name,'Path' : Path})
    return Babyface_df

def EUCpointwise( inputs, targets):
    loss = F.mse_loss(inputs, targets, reduction="none")
    loss = torch.sqrt(torch.sum(loss,axis =2))

    return loss

def exportply(colors, raw_path, out_path, freq, name):
    
    ply = PlyData.read(raw_path)
    
    f = ply.elements[1]
    vertex_data = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    list_v = np.array([ply["vertex"]["x"],ply["vertex"]["y"],ply["vertex"]["z"], colors[:,0]*255, colors[:,1]*255, colors[:,2]*255]).T.tolist()
    list_t = [tuple(e) for e in list_v]
    vertices_with_color = np.array(list_t,dtype = vertex_data )
    vertex_element = PlyElement.describe(vertices_with_color, 'vertex')
    
    modified_ply = PlyData([vertex_element,f],text = False)
    
    new_path = out_path+"/"+str(name)+"_"+str(freq)+".ply"
    modified_ply.write(new_path)
    
def meanply(Data,path):
    
        i = 0
        for idx in Data.index:
            raw_path = Data.Path[idx]
            ply = PlyData.read(raw_path)
            vertex=ply["vertex"]
            verts_init = np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)
            verts_init = verts_init.astype('float32')
    
            if i==0:
                mean=verts_init
                i=i+1
            else:
                mean=mean+verts_init
          
        m = mean/len(Data)
        
        ply["vertex"]["x"] = m[:,0]
        ply["vertex"]["y"] = m[:,1]
        ply["vertex"]["z"] = m[:,2]
        
        ply.write(path)
        
        return(ply)
    
def evaluate(folder,Babyface_df_test):
    Babyface_df_test = create_df(folder,True)
    procrustes_ply(Babyface_df_test,folder+"\\procrustes")
    Babyface_df_test = create_df(folder+"\\procrustes")
    Babyface_df_test.drop([63])
    
    mean_test = meanply(Babyface_df_test,out_path +"\\mean.ply")
    
    mean = np.load(r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\mean_BBF.npy")
    test_set = autoencoder_dataset(root_dir = root_dir, points_dataset = Babyface_df_test, mean=mean,dummy_node = False, mode ="test", typ = "BBF")
    testloader = DataLoader(test_set,batch_size=1)
    
    result_df = pd.DataFrame(columns=["nb_freq","MAE_TEST","EUC_TEST"]) 
    i = 0
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=m, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    
    fig, ax = plt.subplots(figsize=(1, 20))
    cbar = fig.colorbar(mapper,cax=ax, orientation='vertical')
    cbar.set_label('EUC (mm)', size=40)  # Set label size
    cbar.ax.tick_params(labelsize=35)  # Set tick label size
    plt.show()
    
    for fr in [85]:
        if config["depth"] == 0:
            config["nb_freq"] = [fr]
        else:
            config["nb_freq"].append(fr)
            
        f = 31027
        outfile_vec = path_decomp +"\\"+ str(f) +"eig_vecs.npy"
        outfile_val = path_decomp +"\\"+ str(f) +"eig_vals.npy"
        
        eig_vecs = np.load(outfile_vec)
        eig_vals = np.load(outfile_val)
        
        freq = max(config["nb_freq"])
        
        eig_vals = eig_vals[:freq]
        eig_vecs = eig_vecs[:,:freq]
        
        model = SDL_skip(config,eig_vecs,eig_vals)
        
        
        PATH = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Weights\SDL_skip\deep0 Paper"+"\\"+str(fr)+"_SDL_skip.pt"
        # Load the state dict from file
        state_dict = torch.load(PATH)
    
        # Load the state dict into the model
        model.load_state_dict(state_dict["model_state_dict"])
        
        wd = model.SkipBlocksDown[0].weights[1].cpu().detach().numpy()
        wu = model.SkipBlocksUp[0].weights[1].cpu().detach().numpy()
        
        model.eval()
        model.cuda()
        criterion = torch.nn.L1Loss()
        criterion2 = EUCLoss()
        running_loss = 0.0
        running_loss2 = 0.0 
        
        EUCP = []
        with torch.no_grad():
            j = 0
            for data in testloader:  # Iterate in batches over the training dataset.
            
                 ls = model.encoder(data['points'].to('cuda')).cpu().detach().numpy()
                 lsp = np.matmul(eig_vecs.T,data['points'].numpy())
                 lss = (ls - wd*lsp)/(1-wd)
                 out = model(data['points'].to('cuda'))  # Perform a single forward pass.
                 outsp = np.matmul(eig_vecs,ls)
                 outs = (out.cpu().detach().numpy() - wu*outsp)/(1-wu)
                 
                 loss = criterion(out, data['points'].cuda())  # Compute the loss.
                 loss2 = criterion2(out, data['points'].cuda())
                 #print(Babyface_df_test.Name[j]," : ",loss2.item())
                 
                 EUC = EUCpointwise(out, data['points'].cuda()).cpu().numpy().flatten()
                 EUCP.append(EUC)
                 colors=mapper.to_rgba(EUC)
                 
                 raw_path = Babyface_df_test.Path[j]
                 name = Babyface_df_test.Name[j]
                 
                 exportply(colors, raw_path, out_path, fr, name)
    
    
    
                 
                 running_loss += loss.item()*len(data["points"])
                 running_loss2 += loss2.item()*len(data["points"])
                 del out, loss, loss2
                 torch.cuda.empty_cache()
                 j+=1
           
        epoch_loss = running_loss/len(testloader.dataset)
        epoch_loss2 = running_loss2/len(testloader.dataset)
        
        
        mEUCP = np.mean(np.array(EUCP),axis=0)
        print(max(mEUCP))
        colors=mapper.to_rgba(mEUCP)
        exportply(colors, out_path +"\\mean.ply", out_path, fr, "mean")
        result_df.loc[i] = [config["nb_freq"][-1],epoch_loss, epoch_loss2]
        print(result_df)
        config["nb_freq"] = config["nb_freq"][:-1]
        i+=1
    
    os.chdir(out_path)
    Name=[]    
    for file in glob.glob("*mean*.ply"):
        Name.append(file.partition(".")[0])
        
    Path=[out_path +"\\"+name+".ply" for name in Name]
    Name.insert(0,"Synthetic_3D_face_00001")
    Path.insert(0,example_path)
        
    mean_df = pd.DataFrame({'Name' : Name,'Path' : Path})
    #important delete old files before
    procrustes_ply(mean_df,out_path)
    
    
        