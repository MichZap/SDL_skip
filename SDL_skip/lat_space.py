# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:26:30 2024

@author: Michael
"""

#code to analyse latent space

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from plyfile import PlyData
from Dataloader import autoencoder_dataset
from torch.utils.data import DataLoader
from SDL import SDL_skip
from eval_adddata import create_df
from Sym_transform import export_ply
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Settings
config = {
  "depth": 2,                   # model depth 0 pools directly to the latent size 
  "nb_freq": [4096,1024,256,170],   # number of eigenvectors considered, here i,m and j from illustration
  "device": "cuda",
  "hidden_dim": [64]*7,         # see d in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "conv_size": [1024]*7,        # see c in illustration, can be varied by block, number of blocks = sum_{i=0}^{depth}2**i
  "prior_coef": 0.4,            # weight w initialization for the skip connection, the learnable component gets scaled by 1-w
  "use_activation":0,           # 0 no activation functions, 1 between every two linear layers, 2 after every linear layer, -1 like 1 but only in the convolution operation
  "dropout":0,              
  "activation": "ELU",          # ReLU, Tanh, Sigmoid, LeakyReLU, ELU, SiLU, SwiGLU 
  "use_conv": True,             # use linear(4,c) and linear(c,3) from illustration
  "use_eigenvals":True,         # use eigenvalue encoding
  "use_norm":False,             # use LayerNorm before every block of linear layers
  "use_att":False,              # use Attention-Matrix for pooling
  "use_norm_att":False,          # in case of use_att True determine if the attentionscore gets normalized
  "sqrt":False,                  # in case of use_att True determine if the attentionscore gets scaled by sqrt of dim
  "n_heads":0,                  # in case of use_att True determine the number of heads, n_heads = 0 and n_heads = 1 slightly different implementation
  "value":False,                 # in case of multihead attention, use additional 3x3 value matrix
  "bias":False,                 # use bias for linear layers
  "p":0,                        # augmentation probability for flipping and rotation per axis
  "reset":True,                 # use axvier uniform weight initialization instead of pytorch default
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
  "sym":False,               

}

## folder where .pt-files will be stored 
root_dir = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\SDL'
## folder containing test ply-files
folder = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\addTestData\procrustes'
folder = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\old+new'
## path to a procrustes alligned ply file from the training data
path_train = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\uu5t1_LSCM_fix573_190903120253_R_FNN_woHat_procr.ply"
## folder containing eigenvalues and eigenvectors
path_decomp = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP'

m_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\mean_BBF.npy"
mat_path = r"C:\Users\Michael\Downloads\T8_SymmetrizedTemplate.mat"
out_path = r'C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Output\SDL_skip'
example_path = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data_Celi_3DGCN\Data\Synthetic_3D_face_00001.ply"


mean = np.load(m_path)

freq = max(config["nb_freq"])
fr = config["nb_freq"][-1]
f = 31027

outfile_vec = path_decomp +"\\"+ str(f) +"eig_vecs.npy"
outfile_val = path_decomp +"\\"+ str(f) +"eig_vals.npy"
eig_vecs = np.load(outfile_vec)
eig_vals = np.load(outfile_val)
eig_vals = eig_vals[:freq]
eig_vecs = eig_vecs[:,:freq]

model = SDL_skip(config,eig_vecs,eig_vals)
PATH = r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Weights\SDL_skip\deep2 Paper"+"\\"+str(fr)+"_SDL_skip.pt"
# Load the state dict from file
state_dict = torch.load(PATH)

# Load the state dict into the model
model.load_state_dict(state_dict["model_state_dict"])

model.eval()
model.cuda()


def interpolate(Lat_space,i,j,alpha,model,ply,df,out_path,fr,mean):
    inter = torch.tensor(Lat_space[i]*alpha+Lat_space[j]*(1-alpha)).unsqueeze(0).cuda()
    out = model.decoder(inter).detach().cpu().numpy()[0] + mean
    name = df.Name[i] +"_"+ str(alpha)+"_"+ df.Name[j]
    path = out_path+"/"+str(name)+"_"+str(fr)+".ply"
    export_ply(ply,path,out)

def Latent_space(model, mean, folder,root_dir,plot = False,sym = False, mat = None):
    Babyface_df_test = create_df(folder,False, sym,mat)
    #Babyface_df_test.drop([63])
        

    test_set = autoencoder_dataset(root_dir = root_dir, points_dataset = Babyface_df_test, mean=mean,dummy_node = False, mode ="test", typ = "BBF")
    testloader = DataLoader(test_set,batch_size=1)
    
    Lat_space = []
    with torch.no_grad():

        for data in testloader:  # Iterate in batches over the training dataset.
            lat = model.encoder(data['points'].to('cuda')).cpu().numpy()[0]
            Lat_space.append(lat)
            if plot:
                ax = plt.axes(projection='3d')
                zdata = lat[:,2]
                xdata = lat[:,0]
                ydata = lat[:,1]
                ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
            
                plt.show()
        
        Lat_space = np.array(Lat_space)
        if plot:
            for k in range(Lat_space.shape[1]):
                ax = plt.axes(projection='3d')
                zdata = Lat_space[:,k,2]
                xdata = Lat_space[:,k,0]
                ydata = Lat_space[:,k,1]
                ax.scatter3D(xdata, ydata, zdata, c=zdata)
                plt.show()
            for k in range(Lat_space.shape[1]):
                
                plt.hist(Lat_space[:,k,2])
                plt.show()
                plt.hist(Lat_space[:,k,0])
                plt.show()
                plt.hist(Lat_space[:,k,1])
                plt.show()
                plt.plot(Lat_space[:,k,0],Lat_space[:,k,1])
                plt.show()
        return Lat_space, Babyface_df_test

Lat_space, Babyface_df_test = Latent_space(model, mean, folder,root_dir,plot = False,sym = config["sym"], mat = mat_path)

def convergence(model, mean, folder,root_dir, mat = None):
    Corr = []
    for sym in [False,"sym","asym"]:
        Lat_space, _ = Latent_space(model, mean, folder,root_dir,plot = False,sym = sym, mat = mat)
        Lat_space_flat = Lat_space.reshape(Lat_space.shape[0],-1)
        corr = np.corrcoef(Lat_space_flat, rowvar=False)
        sb.heatmap(corr, cmap="PiYG").set_title(sym)
        plt.show()
        Corr.append(corr)
    corr_diff_sym = Corr[1] - Corr[0]
    corr_diff_asym = Corr[2] - Corr[1]
    sb.heatmap(corr_diff_sym, cmap="PiYG").set_title("diff_sym")
    plt.show()
    sb.heatmap(corr_diff_asym, cmap="PiYG").set_title("diff_asym")
    plt.show()
    
    return np.mean(abs(corr_diff_sym)),np.mean(abs(corr_diff_asym))
           
    
#plot correlation
# Lat_space_flat = Lat_space.reshape(Lat_space.shape[0],-1)
# corr = np.corrcoef(Lat_space_flat, rowvar=False)
# sb.heatmap(corr, cmap="PiYG")
# plt.show()
# mask = abs(corr)>0.1
# sb.heatmap(corr*mask, cmap="PiYG")
# plt.show()

#convergence(model, mean, folder,root_dir, mat = mat_path)
        
#test interpolation
ply = PlyData.read(example_path)
# for alpha in np.arange(-1, 2.25, 0.25):
#     interpolate(Lat_space,0,10,alpha,model,ply,Babyface_df_test,out_path,fr,mean)
        

        
def sample(n_samples,model,ply,mean,array,out_path,fr,flatten = False,use_mask = False,thres = 0,GCV=False,alpha = 1000, max_iter = 100, mode = "cd"):
    

    Sampled_Latent = []
    
    if flatten:
        rarray = array.reshape(array.shape[0],-1)
        if GCV:
            cov = GraphicalLasso(alpha=alpha,max_iter=max_iter,mode=mode).fit(rarray)
            cov = cov.covariance_
        else:
            cov = np.cov(rarray, rowvar=False)
            if use_mask:
                corr = np.corrcoef(rarray, rowvar=False)
                sd = np.std(rarray,axis = 0)
                diag_sd = np.diag(sd)
                mask = abs(corr)>thres
                cov = diag_sd @ corr*mask @ diag_sd

        m = np.mean(rarray, axis = 0)
        
        
        for name in range(n_samples):
            rvec = np.random.multivariate_normal(m,cov).reshape(-1,3)
    
            SL_t = torch.tensor(rvec).unsqueeze(0).cuda().float()
            out = model.decoder(SL_t).detach().cpu().numpy()[0] + mean
            
            path = out_path+"/"+str(name)+"_"+str(fr)+"_"+str(flatten)+"_"+str(use_mask)+str(thres)+str(GCV)+str(alpha)+str(max_iter)+str(mode)+".ply"
            export_ply(ply,path,out)

    else:
        m = np.mean(array, axis = 0)

        for name in range(n_samples):
            for i in range(array.shape[1]):
                cov = np.cov(array[:,i,:], rowvar=False)
                rvec = np.random.multivariate_normal(m[i],cov)
                Sampled_Latent.append(rvec)
            SL = np.array(Sampled_Latent)
            SL_t = torch.tensor(SL).unsqueeze(0).cuda().float()
            out = model.decoder(SL_t).detach().cpu().numpy()[0] + mean
            
            path = out_path+"/"+str(name)+"_"+str(fr)+"_"+str(flatten)+"_"+str(use_mask)+str(thres)+str(GCV)+str(alpha)+str(max_iter)+".ply"
            export_ply(ply,path,out)

    

sample(100,model,ply,mean,Lat_space,out_path,fr,True,False,0,True,alpha=175,max_iter = 10000,mode = "cd")


