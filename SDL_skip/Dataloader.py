from torch.utils.data import Dataset
import torch
import os
import glob
from plyfile import PlyData
import numpy as np
import pandas as pd
from utilis import meanply
from Transforms import RandomFlip, RandomRotate
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split

def load_data(p,root_dir,folder,axis, typ ="Coma", flip = True, rot = True):
     
    rot_t = [RandomRotate(axis=ax,p=p)   for ax in axis] if rot else None
    flip_t = [RandomFlip(axis=ax,p=p)   for ax in axis] if flip else None

    if rot and flip:
        transform = Compose([Compose(rot_t),Compose(flip_t)])
    if rot and not flip:
        transform = Compose(rot_t)
    if not rot and flip:
        transform = Compose(flip_t)
    
    if not rot and not flip:
        transform = None

    
    if typ == "FWH":
        df_train = np.load(folder+"//"+"train_data.npy", mmap_mode='r')
        df_val = np.load(folder+"//"+"test_data.npy", mmap_mode='r')
    elif typ == "Coma":
        Name = []
        Test = []
        Path = []
        Face = []
        
        k=0
        for i in os.scandir(folder):
            if i.is_dir():
                for j in os.scandir(i):
                    if j.is_dir():
                        for file in os.listdir(j):
                            if file.endswith(".ply"):
                                Path.append(j.path+"/"+file)
                                Name.append(file.partition(".ply")[0])
                                Face.append(i.name)
                                if k%100<10:
                                    Test.append(1)
                                else:
                                    Test.append(0)
                                k=k+1
            
        
        Coma_df=pd.DataFrame({'Name' : Name,'Test':Test,'Face':Face,'Path' : Path})
        Coma_df.Test = (Coma_df.Face == "FaceTalk_170913_03279_TA") | (Coma_df.Face == "FaceTalk_170915_00223_TA")
        
        df_train = Coma_df[Coma_df.Test==False]
        df_val = Coma_df[Coma_df.Test==True]
        
        
    
    else:                     
        #create df
        Name=[]

        os.chdir(folder)
        for file in glob.glob("*.ply"):
            Name.append(file.partition(".")[0])

        Path=[folder +"\\"+name+".ply" for name in Name]

        Babyface_df=pd.DataFrame({'Name' : Name,'Path' : Path})

        np.random.seed(31415)
        torch.manual_seed(31415)

        df_train,df_val = train_test_split(Babyface_df,test_size = 0.2)
                         
    mean=meanply(df_train, typ = typ)  
        
    train_set = autoencoder_dataset(root_dir = root_dir, points_dataset = df_train, mean=mean,transform=transform,dummy_node = False, mode="train",typ = typ)
    val_set = autoencoder_dataset(root_dir = root_dir, points_dataset = df_val, mean=mean,dummy_node = False, mode ="val", typ = typ)

    return train_set, val_set

#based on Neural3DMM Datalaoder
class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, mean=None, transform=None, dummy_node = True, mode="train", typ = "Coma"):
        
        self.mean = mean
        self.transform = transform
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.dummy_node = dummy_node
        self.typ = typ
        self.mode = mode
        
    def __len__(self):
        return len(self.points_dataset)

    def __getitem__(self, idx):


        os.makedirs(self.root_dir+"\\processed",exist_ok = True)
        if self.typ == "FWH":
            path=self.root_dir+"\\processed\\"+self.mode+"_"+str(idx)+".pt"
        elif self.typ == "Coma":
            path=self.root_dir+"\\processed\\"+self.points_dataset.Face.iloc[idx]+"_"+self.points_dataset.Name.iloc[idx]+".pt"
        else:
            path=self.root_dir+"\\processed\\"+self.points_dataset.Name.iloc[idx]+".pt"
        
        if os.path.isfile(path):
            verts = torch.load(path)
        
        else:
            if self.typ == "FWH":
                vertex = self.points_dataset[idx]
                n= int(len(vertex)/3)
   
                verts_init=np.stack((vertex[:n], vertex[n:2*n], vertex[2*n:3*n]),axis=1)
            else:
                raw_path = self.points_dataset.Path.iloc[idx]
                ply = PlyData.read(raw_path)
                vertex=ply["vertex"]
                verts_init=np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)  

            sc = 1000 if self.typ == "Coma" else 1        
            verts_init = verts_init.astype('float32')*sc
            
            if self.mean is not None:
                verts_init = verts_init -self.mean

            if self.dummy_node:
                #adds an additional vetex [0,0,0] add the end
                verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
                verts[:-1,:] = verts_init
                verts_init = verts 

            verts = torch.Tensor(verts_init)
            torch.save(verts,path)
        
        if self.transform:
            verts = self.transform(verts)
            
        sample = {'points': verts}

        return sample
