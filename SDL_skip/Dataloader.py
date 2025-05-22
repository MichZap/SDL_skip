from torch.utils.data import Dataset
import torch
import os
import glob
from plyfile import PlyData
import numpy as np
import pandas as pd
import random
from utilis import meanply, sdply, identify_subject
from Transforms import RandomFlip, RandomRotate, AddGaussianNoise, SpectralInterpolation
from torchvision.transforms import Compose, RandomChoice
from sklearn.model_selection import train_test_split
from Sym_transform import symtrans, aysmtrans

def trans(batch,p,axis, flip = True, rot = True):
     
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
    return transform(batch)

def load_data(p,degrees,root_dir,folder,mat_path,axis, typ ="Coma", sym = False, std = False, norm = False, p_asym = 1, k = 6, flip = True, rot = True, sigma= False, aug_mean = False, comp = 0,phi = None, unique = False):
     
    
    if typ == "FWH":
        #df_train = np.load(folder+"//"+"train_data.npy", mmap_mode='r')
        #df_val = np.load(folder+"//"+"test_data.npy", mmap_mode='r')
        df_train = np.load(os.path.join(folder, "train_data.npy"), mmap_mode='r')
        df_val = np.load(os.path.join(folder, "test_data.npy"), mmap_mode='r')
        
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
                                #Path.append(j.path+"/"+file)
                                Path.append(os.path.join(j.path,file))
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
        
    elif typ == "BBF_exp":
        Name = []
        Test = []
        Path = []
        Face = []
        Exp = []
        
        os.chdir(folder)
        for file in glob.glob("*.ply"):
            Name.append(file.partition(".")[0])
        
            
        Path = [folder +"\\"+name+".ply" for name in Name]
        Face = [name[:36] for name in Name]
        Exp = [name[65:67] for name in Name]
        
        Face_un = np.unique(Face)
        
        np.random.seed(31415)
        torch.manual_seed(31415)
        
        Face_train, Face_test = train_test_split(Face_un, test_size = 0.2)
        
        Test = [True if face in Face_test else False for face in Face]
        
        Babyface_df=pd.DataFrame({'Name' : Name,'Test':Test,'Face':Face,'Exp':Exp,'Path' : Path})
        
        df_train = Babyface_df[Babyface_df.Test==False]
        df_val = Babyface_df[Babyface_df.Test==True]
        
    else:                     
        #create df
        Name=[]

        os.chdir(folder)
        for file in glob.glob("*.ply"):
            Name.append(file.partition(".")[0])

        #Path=[folder +"\\"+name+".ply" for name in Name]
        Path = [os.path.join(folder, name + ".ply") for name in Name]
        #Asym = [False for name in Name]

        Babyface_df=pd.DataFrame({'Name' : Name,'Path' : Path}).sort_values(by='Path')
        
        Babyface_df['Subject'] = Babyface_df['Name'].apply(identify_subject)

        np.random.seed(31415)
        torch.manual_seed(31415)

        if typ == "BBF":
            if unique == True:
                # Get unique subjects
                unique_subjects = Babyface_df['Subject'].unique()
                
                # Split the unique subjects into train and test sets
                train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)
                
                # Filter the DataFrame based on the split subjects
                df_train = Babyface_df[Babyface_df['Subject'].isin(train_subjects)]
                df_val = Babyface_df[Babyface_df['Subject'].isin(test_subjects)]
            else:
                df_train,df_val = train_test_split(Babyface_df,test_size = 0.2)
            
        else:
            df_train =  Babyface_df
            df_val = Babyface_df
    
    if sym == "sym" or sym == "asym":
        Name = []
        Path = []
        
        for path in df_train.Path:
            newpaths,newnames = symtrans(path,mat_path)
            Path.extend(newpaths)
            Name.extend(newnames)
            
        df_train_sym = pd.DataFrame({'Name' : Name,'Path' : Path})
        df_train = pd.concat([df_train,df_train_sym] , ignore_index=True)
        
    if sym == "asym":
        Name = []
        Path = []
        
        for path in df_train.Path:
            if not "sym" in path:
                for path1 in df_train.Path:
                        if path != path1 and not "sym" in path1:
                                newpaths,newnames = aysmtrans(path,path1,mat_path)
                                Path.extend(newpaths)
                                Name.extend(newnames)
                            
        #Asym = [True for name in Name]
    
        df_train_asym = pd.DataFrame({'Name' : Name,'Path' : Path})
                
        #df_train = pd.concat([df_train,df_train_asym] , ignore_index=True)
    else:
        df_train_asym = None
    
    if sym == "sym" or sym == "asym":
        df_mean = pd.concat([df_train,df_train_sym] , ignore_index=True)
    else:
        df_mean = df_train
    
    
    if typ == "BBF_add":
        mean = np.load(r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\SAE_LP\procrustes\mean_BBF.npy")
    else:
        mean = meanply(df_mean, typ = typ, sym = sym, SSH = norm)
        sd = sdply(df_mean, typ = typ, sym = sym) if std else None
        
    if sigma > 0:
        sigma_flag = True
    else:
        sigma_flag = False
        
    rot_t = [RandomRotate(axis=ax,p=p)   for ax in axis] if rot == True else []
    flip_t = [RandomFlip(axis=ax,p=p)   for ax in axis] if flip == True else []
    gn_t =  [AddGaussianNoise(sigma,p)] if sigma_flag == True else []
    spec_int = [SpectralInterpolation(df_train, phi, comp, root_dir,p=p,mean = mean, sd = sd, norm = norm, sym = sym)] if comp > 0 else []
    
    if not rot and not flip and not sigma_flag and comp == 0:
        transform = None
        
    else:
        list_transforms = rot_t+ flip_t + gn_t + spec_int
        if k >0:
            random_transforms = [RandomChoice(list_transforms) for _ in range(k)]
        else:
            random_transforms = list_transforms
            
        preprocess_transforms = []
        postprocess_transforms = []
        if aug_mean == True:

            preprocess_transforms.append(lambda x: x + mean)
            postprocess_transforms.append(lambda x: x - mean)
            
        transform = preprocess_transforms + random_transforms + postprocess_transforms
        transform = Compose(transform)
        
    train_set = autoencoder_dataset(root_dir = root_dir, points_dataset = df_train, mean = mean, sd = sd, norm = norm, sym = sym, transform=transform,dummy_node = False, mode="train",typ = typ, p_asym = p_asym, data_asym = df_train_asym )
    val_set = autoencoder_dataset(root_dir = root_dir, points_dataset = df_val, mean = mean, sd = sd, norm = norm, sym = sym, dummy_node = False, mode ="val", typ = typ)

    return train_set, val_set

#based on Neural3DMM Datalaoder
class autoencoder_dataset(Dataset):

    def __init__(self, root_dir, points_dataset, mean=None, sd = None, norm = False, sym = False, transform=None, dummy_node = True, mode="train", typ = "Coma",p_asym = 0, data_asym = None):
        
        self.mean = mean
        self.sd = sd
        self.transform = transform
        self.root_dir = root_dir
        self.points_dataset = points_dataset
        self.df_asym = data_asym
        self.dummy_node = dummy_node
        self.typ = typ
        self.mode = mode
        self.norm = norm
        self.sym = sym
        self.p_asym = p_asym
        
    def __len__(self):
        return len(self.points_dataset)
    
    def load(self,info,idx):
        
        #os.makedirs(self.root_dir+"\\processed",exist_ok = True)
        os.makedirs(os.path.join(self.root_dir, "processed"), exist_ok=True)
        
        
        if self.typ == "FWH":

            path = os.path.join(self.root_dir, "processed", f"{self.mode}_{idx}.pt")
        elif self.typ == "Coma":
            
            path = os.path.join(self.root_dir, "processed", f"{info.Face}_{info.Name}.pt")
        else:

            path = os.path.join(self.root_dir, "processed", f"{info.Name}.pt")
            
        if self.mean is not None:
            path = path[:-3] + "_mean.pt"
        
        if self.sd is not None:
            path = path[:-3] + "_sd.pt"
            
        if self.norm:
            path = path[:-3] + "_norm.pt"
            
        if self.sym == "sym" or self.sym == "asym":
            path = path[:-3] + "_sym.pt"
            
        
        if os.path.isfile(path):

            sample = torch.load(path)
            verts = sample["points"]
        
        else:
            if self.typ == "FWH":
                vertex = self.points_dataset[idx]
                n= int(len(vertex)/3)
   
                verts_init=np.stack((vertex[:n], vertex[n:2*n], vertex[2*n:3*n]),axis=1)
            else:
                raw_path = info.Path
                ply = PlyData.read(raw_path)
                vertex=ply["vertex"]
                verts_init=np.stack((vertex['x'], vertex['y'], vertex['z']),axis=1)  

            sc = 1000 if self.typ == "Coma" else 1        
            verts_init = verts_init.astype('float32')*sc
            
            SSH = 1
            if self.norm:
                SSH = np.sqrt(((verts_init - verts_init.mean(0))**2).sum())/verts_init.shape[0]
                verts_init = verts_init/SSH
            
            if self.mean is not None:
                verts_init = verts_init -self.mean
                
            if self.sd is not None:
                verts_init = verts_init/self.sd
 

            if self.dummy_node:
                #adds an additional vetex [0,0,0] add the end
                verts = np.zeros((verts_init.shape[0]+1,verts_init.shape[1]),dtype=np.float32)
                verts[:-1,:] = verts_init
                verts_init = verts 

            verts = torch.Tensor(verts_init)
            
            sample = {'points': verts}
            if self.norm:
                sample["SSH"] = torch.Tensor([SSH])         
            
            torch.save(sample,path)
        
        if self.transform:
            verts = self.transform(verts)
            sample['points'] = verts

        return sample

    def __getitem__(self, idx):


        if random.random() < self.p_asym:
                idx = random.randint(0, len(self.df_asym) - 1)
                info = self.df_asym.iloc[idx]

        else:
                info = self.points_dataset.iloc[idx]


        sample = self.load(info,idx)

        return sample
