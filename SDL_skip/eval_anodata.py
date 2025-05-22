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
from SDL import SDL_skip, Spectral_decomp
from Sym_transform import symtrans, aysmtrans

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Settings
config = {
    
    #learning settings:
    "device": "cuda",
    "learning_rate": 1e-3/2,
    "train_batch_size": 16,
    "test_batch_size": 16,
    "epochs":10000,
    "lambda":0,                   # additonal loss function if FFW == True for spectral latent denoiseing
    
    #learning rate shedule settings:
    "shed_fac":0.75,              # learning rate scale factor on plateau, see X in paper
    "print_epoch":25,
    "shed_pat":25,               # learning rate sheduler patience, see N in paper
    "es_pat":100,                 # early stopping patience, see M in paper
    "min_delta":0.005,                # parameter for early stopping tolerance
    
    #model base settings:
    "depth": 2,                   # model depth 0 pools directly to the latent size 
    "nb_freq": [4096,1024,256],                # number of eigenvectors considered, here i,m and j from illustration
    "hidden_dim": [64]*8,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
    "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
    "use_eigenvals":True,         # use eigenvalue encoding
    "conv_size": [1048]*8,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
    "prior_coef": 0.8,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
    
    #model additional settings:
    "bias":False,                 # use bias for linear layers
    "dropout":0,
    "reset":False,                # use axvier uniform weight initialization instead of pytorch default
    "aw":True,                    # use additional weight to scale every SkipBlock
    "wdim":True,                  # advanced aggregation if True one weight by vertex
    "Skip":2,                     # 0 no skip connections, 1 skip connection to end of each block, 2 skip connection to middle and end, 3 skip connection only to middle
    "learn":True,                 # False disables the learnable component of all SkipBlocks apart of the deepest part of the skipping pyramid, reduces number of parameters approx by a factor of 3
    "direct":False,                # if True the second skip conncetion start from the input
    "FFW":False,                   # additonal FFW layer to denosie the latent space for spectral decomp   
    
    #activation settings:
    "use_activation":0,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
    "activation": "ReLU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU   
        
    #attention settings:
    "use_att":False,              # use Attention-Matrix for pooling
    "n_heads":0,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
    "use_norm":False,             # use LayerNorm before every block of linear layers
    "use_norm_att":"Soft",        # in case of use_att True determine if the attentionscore gets normalized
    "sqrt":True,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
    "value":True,                # in case of multihead attention, use additional 3x3 value matrix
    "vinit":True,                 # vertex initialization for attention matrix
    "eb_dim":64,                  # in case of vertex initialization: None => no embedding, else verteices are embedded to eb_dim before attention

    #augmentation settings
    "p":0,                      # augmentation probability for flipping and rotation per axis
    "k":3,                        # max amount of consecutive augmentations
    "deg":180,                     #rotation degree for augmentation
    "flip":True,                 #False disables flipping augmentation
    "add_data":False,             # if we use addition data from folder_add, not considered in train test split
    "sym":"sym",                  # False default, "sym" adds mirrored and symmetric mesh, "aysm" additional transfers asymtries from one face to another requires indices for mirroring
    "p_asym":0.025,                 # probability for asymmetrty transfer to control ratio between real and asymmetric data

    #SDL_simple                    Similar to SDL but no with constant width and advanced aggregation by spectral dim
    "simple":False,                # uses simple model
    "gate":False,                  # for simple version gate the weight parameter if true
    "flatten":False,               # flatten of the hidden dim during pooling only work if use_att false and simpel True
    "proj":True,                   # if False uses weights instead of proj matrix in mh attention
    "Pai": True,                   # Pai conv in the end
    "paiconv_size":[9,9],          # Pai conv size
    "in_ch":[3,3],                 # Pai conv in channels
    "out_ch":[3,3],                # Pai conv out channels
    "pai_act":[1,1],               # Pai conv activation  0 or 1
    "pai_skip":[True,True],
    "pai_small":[False,False],
    
    #additonal settings:
    "typ": "BBF",                 # data to be used "BBF", "Coma" and "FWH"
    "sd":False,                   # sd normalization in Dataloader
    "SSH": True,                 # normalizetion in data loader such that all meshes have norm = 1
    "num_workers": 0,             # num worker for dataloader
    "save_epochs":[0,0,0,296,291,218,227,207,0,0],
}

## folder where .pt-files will be stored 
root_dir = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean\pt"
## folder containing test ply-files
folder = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean"
folder2 = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\Anormal_scans_clean\proc"
## path to a procrustes alligned ply file from the training data
path_train = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean\008_16092014_procr.ply"
## folder containing eigenvalues and eigenvectors
path_decomp = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'

out_path = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\Spectral\anomaly_det2'
example_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\Data\Synthetic_3D_face_00001.ply"
mean_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new_clean\mean_sym_1_SSH_1_BBF.npy"
mat_path = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"

area_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output"

index_LEAR = np.load(area_path + "//left_ear.npy")
index_REAR = np.load(area_path + "//right_ear.npy")
index_EARs = np.union1d(index_REAR, index_LEAR)
index_LEYE = np.load(area_path + "//left_eye.npy")
index_REYE = np.load(area_path + "//right_eye.npy")
index_EYEs = np.union1d(index_REYE, index_LEYE)
index_mouth = np.load(area_path + "//mouth.npy")
index_nose = np.load(area_path + "//nose.npy")
index_all = np.arange(31027)
index_noEARs = np.setdiff1d(index_all, index_EARs)


indices = [index_noEARs,index_EARs,index_LEAR,index_REAR,index_LEYE,index_REYE,index_EYEs,index_mouth,index_nose]


#max error in mm for color coding
m = 3

def create_df(folder,proc = False,sym = False, mat_path = mat_path):
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
        Name.insert(0,"008_16092014_procr")
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
    
def evaluate(folder,out_path,path_decomp,root_dir,config,mean_path,sym=False,spec=False,index = None ):
    
    
    #create df
    Babyface_df_test = create_df(folder,sym = sym)
    

    
    mean = np.load(mean_path)
    meant = torch.tensor(mean,dtype=torch.float32).to('cuda')
    test_set = autoencoder_dataset(root_dir = root_dir, points_dataset = Babyface_df_test, mean=mean,dummy_node = False, mode ="test", typ = "BBF",norm =True,sym="sym")
    testloader = DataLoader(test_set,batch_size=1)
    
    if index != None:
        result_df = pd.DataFrame(columns=["nb_freq","MAE_TEST","EUC_TEST","EUC_noEARs","EUC_EARs","EUC__LEAR","EUC__REAR","EUC_LEYE","EUC_REYE","EUC_EYEs","EUC_mouth","EUC_nose"])
    else:
        result_df = pd.DataFrame(columns=["nb_freq","MAE_TEST","EUC_TEST"])
    i = 0
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=m, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
    
    fig, ax = plt.subplots(figsize=(1, 20))
    cbar = fig.colorbar(mapper,cax=ax, orientation='vertical')
    cbar.set_label('EUC (mm)', size=40)  # Set label size
    cbar.ax.tick_params(labelsize=35)  # Set tick label size
    plt.show()
    
    out_path_o = out_path
    
    frs = [21,42,85,170,341]
    max_fr = max(frs)
    Lat =[]
    Name = []
    EUCL = []
    EUC_reg = []
    
    for fr in frs:
        
        out_path = out_path_o + "\\" + str(fr)
        os.makedirs(out_path,exist_ok = True)
        
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
        
        model = SDL_skip(config,eig_vecs,eig_vals) if spec == False else Spectral_decomp(config,eig_vecs)
        
        
        PATH = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Weights\SDL_skip\deep2 sym SSH"+"\\"+str(fr)+"_SDL_skip.pt"
        # Load the state dict from file
        state_dict = torch.load(PATH)
        
        if spec == False:
            # Load the state dict into the model
            model.load_state_dict(state_dict["model_state_dict"])
        
        model.eval()
        model.cuda()
        criterion = torch.nn.L1Loss()
        criterion2 = EUCLoss()
        running_loss = 0.0
        running_loss2 = 0.0 
        
        EUCP = []
        
        if index is not None:
            num_regions = len(index)
            running_region_losses = [0.0 for _ in range(num_regions)]
        
        with torch.no_grad():
            j = 0
            for data in testloader:  # Iterate in batches over the training dataset.
                 out, outs = model(data['points'].to('cuda'))  # Perform a single forward pass.
                 lat = model.encoder(data['points'].to('cuda')).cpu().flatten().tolist()#lat
                 
                 if 'SSH' in data:
                     SSH = data['SSH'].unsqueeze(-1).to('cuda')
                     outp = (out+meant)*SSH
                     inp = (data['points'].cuda()+meant)*SSH
                 else:
                     outp = out
                     inp = data['points'].cuda()
                     
                 loss = criterion(outp, inp)  # Compute the loss.
                 loss2 = criterion2(outp, inp)
                 EUC = EUCpointwise(outp, inp).cpu().numpy().flatten()
                 
                 
                 if index is not None:
                    RG_LOSS = []
                    for k, region_indices in enumerate(index):
                        # Convert numpy indices to a tensor on the same device as outp.
                        region_idx_tensor = torch.tensor(region_indices,dtype=torch.long, device=outp.device)
                        # Index the predictions and targets for the current region.
                        region_outp = outp[:, region_idx_tensor, :]
                        region_inp = inp[:, region_idx_tensor, :]
                        rg_loss = criterion2(region_outp, region_inp)
                        RG_LOSS.append(rg_loss.item())
                        # Multiply by batchsize
                        running_region_losses[k] += rg_loss.item() * len(data["points"])


                    EUC_reg.append(RG_LOSS)
                    
                 Name.append(Babyface_df_test.Name[j])
                 EUCL.append(loss2.item())
                 Lat.append(lat)  
                 
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
        
        # Compute average loss for each region.
        if index is not None:
            epoch_region_losses = [loss/len(testloader.dataset) for loss in running_region_losses]
        else:
            epoch_region_losses = []
        

        mEUCP = np.mean(np.array(EUCP),axis=0)
        print(max(mEUCP))
        result_df.loc[i] = [config["nb_freq"][-1],epoch_loss, epoch_loss2] + epoch_region_losses
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
            print(result_df)
        config["nb_freq"] = config["nb_freq"][:-1]
        i+=1

    Lat_padded = [lst + [None] * (max_fr*3 - len(lst)) for lst in Lat]
    data = {'Name': Name, 'EUC': EUCL}
    for i in range(max_fr*3):
        data[f'Lat_{i+1}'] = [lst[i] for lst in Lat_padded]
    if index is not None:
        for i, region_indices in enumerate(index):
            data[f'EUC_{i+1}'] = [lst[i] for lst in EUC_reg]
        

    df = pd.DataFrame(data)
    df.to_csv(out_path_o + "\\df4.csv")

    
evaluate(folder,out_path+"\\normal",path_decomp,root_dir,config,mean_path,sym = "sym",spec=True,index = indices)    
evaluate(folder2,out_path+"\\anormal",path_decomp,root_dir,config,mean_path,spec=True,index = indices)         