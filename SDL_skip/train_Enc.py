# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:37:32 2024

@author: Michael
"""

import torch
import numpy as np
import pandas as pd

from Dataloader import load_data
from torch.utils.data import DataLoader
from utilis import EUCLoss

def train_encoder(Encoder, Decoder, Discriminator, config):
    
    print("start Encoder training")
    criterion =  torch.nn.MSELoss()
    criterion2 = EUCLoss()
    
    Encoder.to(config["device"])
    Decoder.to(config["device"])
    Discriminator.to(config["device"])
    
    optimizer_E = torch.optim.Adam(Encoder.parameters(), lr=config["learning_rate"])
    train_set, val_set = load_data(config["p"],config["root_dir"],config["folder"],[0,1,2], config["typ"])
    
    valloader = DataLoader(val_set,batch_size=config["train_batch_size"],shuffle=False)    
    trainloader = DataLoader(train_set,batch_size=config["train_batch_size"],shuffle=True)
    mean = np.load(config["folder"]+"\\mean_"+config["typ"]+".npy")

    list_loss = []
    

    
    for epoch in range(config["epochs"]):
        
         running_loss_1 = 0
         running_loss_2 = 0
         
         Encoder.train()
         Decoder.eval()
         Discriminator.eval()

         for i, data in enumerate(trainloader):
             

             # Configure input
             real_meshes = data['points'].to(config["device"])

             if config["mean"]:
                 real_meshes += torch.tensor(mean).to(config["device"])

                 
             z,_ = Encoder(real_meshes)

             out = Decoder(z)

             _, out_feat = Discriminator(out)
             _, org_feat = Discriminator(data['points'].cuda())
             
             loss = criterion2(out, data['points'].cuda()) + config["k"]*criterion(out_feat,org_feat)  # Compute the loss.
             loss.backward()  # Derive gradients.
             optimizer_E.step()  # Update parameters based on gradients.
             for param in Encoder.parameters():
                param.grad = None  # Clear gradients.
                
             loss2 = criterion2(out, data['points'].cuda())

             running_loss_1 += loss.item()*len(data["points"])
             running_loss_2 += loss2.item()*len(data["points"])
             
             del out, loss, loss2
             torch.cuda.empty_cache()
            
         l1train = running_loss_1/len(trainloader.dataset)
         l2train = running_loss_2/len(trainloader.dataset)

         
         torch.cuda.empty_cache()
         Encoder.eval()
         
         running_loss_1 = 0
         running_loss_2 = 0
         
         with torch.no_grad():
             for data in valloader:  # Iterate in batches over the training dataset.
             
                real_meshes = data['points'].to(config["device"])
                if config["mean"]:
                    real_meshes += torch.tensor(mean).to(config["device"])
                    
                z,_ = Encoder(real_meshes)
                out = Decoder(z)
                _, out_feat = Discriminator(out)
                _, org_feat = Discriminator(data['points'].cuda())
                
                loss = criterion2(out, data['points'].cuda()) + config["k"]*criterion(out_feat,org_feat)  # Compute the loss.
                loss2 = criterion2(out, data['points'].cuda())
                
                running_loss_1 += loss.item()*len(data["points"])
                running_loss_2 += loss2.item()*len(data["points"])
                
                del out, loss, loss2
                torch.cuda.empty_cache()
            
         epoch_loss = running_loss_1/len(valloader.dataset)
         epoch_loss2 = running_loss_2/len(valloader.dataset)
         
         list_loss.append([epoch_loss,epoch_loss2,l1train,l2train])
         
         print("epoch:",epoch,"Loss_val:", epoch_loss, "EUC_loss_val:", epoch_loss2,"Loss_train:", l1train, "EUC_loss_train:", l2train)
 
    df_loss = pd.DataFrame(data = list_loss, columns=["Loss_val","EUC_loss_val","Loss_train","EUC_loss_train"])
    df_loss.plot(subplots=True)