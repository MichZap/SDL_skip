# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:12:02 2024

@author: Michael
"""
import torch.autograd as autograd
import pandas as pd
import numpy as np
import torch
from Dataloader import load_data, trans
from torch.utils.data import DataLoader
from plyfile import PlyData


def compute_gradient_penalty(D,real_data, generated_data,device):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = autograd.Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated,_ = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch

    gradients = gradients.reshape(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()

def train_wgangp(config, generator, discriminator,thres):
   

   if config["ada"]:
       p = 0
   else:
       p = config["p"]
   
   lambda_gp = config["lambda"]
   list_loss = []
   list_TF = []

   generator.to(config["device"])
   discriminator.to(config["device"])

   optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"])
   optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=config["learning_rate"])
   
   train_set, val_set = load_data(p,config["root_dir"],config["folder"],config["mat_path"],[0,1,2], config["typ"],config["sym"])
   
   mean = np.load(config["folder"]+"\\mean_"+config["typ"]+".npy")
    
        
   valloader = DataLoader(val_set,batch_size=config["train_batch_size"],shuffle=False)    
   dataloader = DataLoader(train_set,batch_size=config["train_batch_size"],shuffle=True)
   
   r_t = 0
   
   for epoch in range(config["epochs"]+1):
       
        print("======= Epoch: ", epoch," ========")
        
        running_loss_d = 0
        running_loss_g = 0
        running_loss_n = 0
        running_loss_gp = 0
        for i, data in enumerate(dataloader):

            # Configure input
            real_meshes = data['points'].to(config["device"])
            real_meshes = trans(real_meshes,p,[0,1,2]) if config["ada"] else real_meshes
            
            if config["mean"]:
                real_meshes += torch.tensor(mean).to(config["device"])

            # ---------------------
            #  Train Discriminator
            # ---------------------
            discriminator.train()
            optimizer_D.zero_grad()

            # Sample noise as generator input

            z = torch.randn((real_meshes.shape[0],config["nb_freq"][-1],3), device=config["device"])


            # Generate a batch of images
            fake_meshes = generator(z)
            fake_meshes = trans(fake_meshes,p,[0,1,2])

            # Real images
            real_validity,_ = discriminator(real_meshes)

            # Fake images
            fake_validity,_ = discriminator(fake_meshes.detach())
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator,
                                                        real_meshes,
                                                        fake_meshes,
                                                        config["device"])
            # Adversarial loss
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                      + lambda_gp * gradient_penalty)
            gp = lambda_gp * gradient_penalty
            
            d_loss.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            
            r_tt = torch.sum(real_validity > thres) - torch.sum(real_validity <= thres)

            
            # Train the generator and output log every n_critic steps
            if i % config["n_critic"] == 0:

                # -----------------
                #  Train Generator
                # -----------------
                generator.train()
                # Sample noise as generator input
                z = torch.randn((real_meshes.shape[0],config["nb_freq"][-1],3), device=config["device"])
                # Generate a batch of images
                fake_meshes = generator(z)
                fake_meshes = trans(fake_meshes,p,[0,1,2])

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity,_ = discriminator(fake_meshes)
                

                norm = torch.mean(torch.norm(fake_meshes,dim=(1, 2)))


                g_loss = -torch.mean(fake_validity) - config["Tau"]*norm

                g_loss.backward()
                optimizer_G.step()
                
                running_loss_g += g_loss.item()*len(data["points"])
                running_loss_n += norm.item()*len(data["points"])
                
            running_loss_d += d_loss.item()*len(data["points"])
            running_loss_gp += gp.item()*len(data["points"])
            r_t += r_tt.item() 
        
            
        epoch_loss_d = running_loss_d/len(dataloader.dataset)
        epoch_loss_g = running_loss_g/(len(dataloader.dataset)/config["n_critic"])
        epoch_loss_n = running_loss_n/(len(dataloader.dataset)/config["n_critic"])
        epoch_loss_gp = running_loss_gp/len(dataloader.dataset)
        epoch_loss_wgan = (running_loss_d - running_loss_gp)/len(dataloader.dataset)
        r_t = r_t/len(dataloader.dataset)
        
        
        report = True if epoch == config["epochs"] else False
        export = True if epoch % 50 ==0 else False        

        epoch_loss_d_val,epoch_loss_g_val, TF = validate(config, generator, discriminator,valloader,epoch,mean,thres,p,report,export)
        list_loss.append([epoch_loss_d,epoch_loss_d_val,epoch_loss_gp,epoch_loss_wgan,epoch_loss_g,epoch_loss_g_val,epoch_loss_n,p])
        
        list_TF.append(TF)
        
        print("p: ", np.round(p,2), "r_t: ",np.round(r_t,2))
        print("WGAN: ", np.round(epoch_loss_wgan,2))
        print("D_loss: ", np.round(epoch_loss_d,2),"D_loss_val: ",np.round(epoch_loss_d_val,2))
        print("G_loss: ", np.round(epoch_loss_g,2),"G_loss_val: ",np.round(epoch_loss_g_val,2))
        print("G_norm: ",np.round(epoch_loss_n,2))
        
        if config["ada"]:
            if r_t >= config["target_value"]:
                p = min(1.0, p + 0.05)
            else:
                p = max(0.0, p - 0.05)
        
   df_loss = pd.DataFrame(data = list_loss, columns=["Disc","Disc_val","GP","WGAN","Gen","Gen_val","norm","p"])
   df_loss.plot(subplots=True)
   
   df_TF = pd.DataFrame(data = list_TF, columns=["FP","TN","TP","FN"])
   df_TF.plot(subplots=True)
   

   return df_loss        
        

def validate(config, generator, discriminator,dataloader,epoch,mean,thres,p,report = False, export = True):
    
    generator.eval()
    discriminator.eval()
    
    lambda_gp = config["lambda"]
    
    real_positives = 0
    false_negatives = 0
    false_positives = 0
    real_negatives = 0
    
    running_loss_d = 0
    running_loss_g = 0
    
    real_list = []
    false_list = []
    
    for i, data in enumerate(dataloader):
        
        real_meshes = data['points'].to(config["device"])
        real_validity,_ = discriminator(real_meshes)
        real_list.extend(real_validity.detach().cpu().tolist())

        
        z = torch.randn((real_meshes.shape[0],config["nb_freq"][-1],3), device=config["device"])


        # Generate a batch of images
        fake_meshes = generator(z)
        
        
        ply = PlyData.read(config["example_path"]) 
        if config["mean"]:
            mean = np.zeros_like(mean)
        if export:
            if i == 0:
                exportply(fake_meshes.detach().cpu().numpy(),config["outpath"],ply,mean,epoch,i)
        fake_meshes = trans(fake_meshes,p,[0,1,2])
        
        
        # Real images
        real_validity,_ = discriminator(real_meshes)
        real_positives += np.sum(real_validity.detach().cpu().numpy()>thres)
        false_negatives += np.sum(real_validity.detach().cpu().numpy()<thres)

        # Fake images
        fake_validity,_ = discriminator(fake_meshes)
        false_positives += np.sum(fake_validity.detach().cpu().numpy()>thres)
        real_negatives += np.sum(fake_validity.detach().cpu().numpy()<thres)
        
        gradient_penalty = compute_gradient_penalty(discriminator,
                                                    real_meshes,
                                                    fake_meshes,
                                                    config["device"])
        # Adversarial loss
        d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                  + lambda_gp * gradient_penalty)
        
        norm = torch.mean(torch.norm(fake_meshes,dim=(1, 2)))


        g_loss = -torch.mean(fake_validity) - config["Tau"]*norm
        
        running_loss_d += d_loss.item()*len(data["points"])
        running_loss_g += g_loss.item()*len(data["points"])
        
        false_list.extend(fake_validity.detach().cpu().tolist())

    
    epoch_loss_d = running_loss_d/len(dataloader.dataset)
    epoch_loss_g = running_loss_g/len(dataloader.dataset)
        
    print("======True==False==")
    print("= Gen: ",false_positives,"   ",real_negatives,"  =")
    print("= Real:",real_positives,"   ",false_negatives,"  =")
    print("===================")
    
    if report:
        print("False:")
        print(false_list)
        print("True:")
        print(real_list)
    
    TF =[false_positives,real_negatives,real_positives,false_negatives]
    
    return epoch_loss_d, epoch_loss_g, TF
        
    
def exportply(meshes,path,ply,mean,epoch,k):
    batch_size = meshes.shape[0]
    for j in range(batch_size):
        array = meshes[j]
        ply["vertex"]["x"] = array[:,0] + mean[:,0]
        ply["vertex"]["y"] = array[:,1] + mean[:,1]
        ply["vertex"]["z"] = array[:,2] + mean[:,2]
        new_path = path+"\\"+str(k)+"_"+str(j)+"_Epoch_"+str(epoch)+" .ply"
        ply=PlyData(ply, text = False)
        ply.write(new_path)   