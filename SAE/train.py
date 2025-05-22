from utilis import EUCLoss
import torch
import numpy as np
from LearnedPooling_mod import LearnedPooling
from utilis import init_weights
from Dataloader import load_data, autoencoder_dataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

    
def train_bf(config):
    print("________________________________START___________________________________")

    #setup Model and Optimizer
    if config["typ"] == "Coma":
        f = "ComaData_5023"
    elif config["typ"] == "FWH":
        f = 11510
    else:
        f = 31027        
    
    #load eigenvectors and value
    outfile_vec = config["path_decomp"] +"\\"+ str(f) +"eig_vecs.npy"

    eig_vecs = np.load(outfile_vec)

    f = config["nb_freq"]

    eig_vecs = eig_vecs[:,:f]
    phi_Tensor = torch.tensor(eig_vecs).float()
    
    model = LearnedPooling(config)
    model.to(config["device"])
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = config["shed_fac"],patience=config["shed_pat"],min_lr = config["learning_rate"]/100)
    early_stopper = EarlyStopper(patience=config["es_pat"], min_delta=0)
    criterion = torch.nn.L1Loss()
    criterion2 = EUCLoss()
    
    
    
    train_set, val_set = load_data(config["p"],config["root_dir"],config["folder"],[0,1,2],eig_vecs, config["typ"],config["flip"],config["rot"],config["sigma"],config["aug_mean"],config["comp"])

    trainloader = DataLoader(train_set,batch_size=config["train_batch_size"],shuffle=True)
    valloader = DataLoader(val_set,batch_size=config["test_batch_size"])

    best_epoch = 0
    best_loss1_t = 9999
    best_loss2_t = 9999
    best_loss1_v = 9999
    best_loss2_v = 9999
    
    #train
    for epoch in range(1, config["epochs"]):
        
        model.train()
        running_loss = 0.0
        running_loss2 = 0.0
        running_loss_real = 0.0
        running_loss2_real = 0.0
        
        

        
        for data in trainloader:  # Iterate in batches over the training dataset.
             out = model(data['points'].to('cuda'))  # Perform a single forward pass.
             loss = criterion(out, data['points'].cuda())  # Compute the loss.
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             for param in model.parameters():
                param.grad = None  # Clear gradients.
                
             loss2 = criterion2(out, data['points'].cuda())
             
             if epoch%config["print_epoch"]==0:
                 
                 loss_real = criterion(torch.matmul(phi_Tensor.cuda(),out), data['org'].cuda()) # Compute the loss.
                 loss2_real = criterion2(torch.matmul(phi_Tensor.cuda(),out), data['org'].cuda())
                 
                 running_loss_real += loss_real.item()*len(data["points"])
                 running_loss2_real += loss2_real.item()*len(data["points"])

             running_loss += loss.item()*len(data["points"])
             running_loss2 += loss2.item()*len(data["points"])

             torch.cuda.empty_cache()
            
        epoch_loss = running_loss/len(trainloader.dataset)
        epoch_loss2 = running_loss2/len(trainloader.dataset)
        l1train  = epoch_loss
        l2train  = epoch_loss2
        
        if epoch%config["print_epoch"]==0:
            l1train_real  = running_loss_real/len(trainloader.dataset)
            l2train_real  = running_loss2_real/len(trainloader.dataset)
        

        torch.cuda.empty_cache()
        
        model.eval()
        
        running_loss = 0.0
        running_loss2 = 0.0
        running_loss_real = 0.0
        running_loss2_real = 0.0 
        
        with torch.no_grad():
            for data in valloader:  # Iterate in batches over the training dataset.
                 out = model(data['points'].to('cuda'))  # Perform a single forward pass.
                 loss = criterion(out, data['points'].cuda())  # Compute the loss.
                 loss2 = criterion2(out, data['points'].cuda())
                 
                 if epoch%config["print_epoch"]==0:
                     
                     loss_real = criterion(torch.matmul(phi_Tensor.cuda(),out), data['org'].cuda()) # Compute the loss.
                     loss2_real = criterion2(torch.matmul(phi_Tensor.cuda(),out), data['org'].cuda())
                     
                     running_loss_real += loss_real.item()*len(data["points"])
                     running_loss2_real += loss2_real.item()*len(data["points"])
                     


                 running_loss += loss.item()*len(data["points"])
                 running_loss2 += loss2.item()*len(data["points"])
                 
                 
                 torch.cuda.empty_cache()
           
        epoch_loss = running_loss/len(valloader.dataset)
        epoch_loss2 = running_loss2/len(valloader.dataset)
        if epoch%config["print_epoch"]==0:
            epoch_loss_real = running_loss_real/len(valloader.dataset)
            epoch_loss2_real = running_loss2_real/len(valloader.dataset)
        
        
        scheduler.step(epoch_loss)
        if early_stopper.early_stop(epoch_loss):
                    
            break
        
        if epoch%config["print_epoch"]==0:
            print("epoch:",epoch,"MAE_loss_val:", epoch_loss_real, "EUC_loss_val:", epoch_loss2_real,"MAE_loss_train:", l1train_real, "EUC_loss_train:", l2train_real)
            if epoch_loss_real < best_loss1_t:
                best_epoch = epoch
                best_loss1_t = epoch_loss_real
                best_loss2_t = epoch_loss2_real
                best_loss1_v = l1train_real
                best_loss2_v = l2train_real
    
    PATH = str(config["size_latent"])+"_"+str(f)+"_SAE.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
    print("Finished Training")
    
    return [config["size_latent"],epoch, best_epoch, best_loss1_t, best_loss2_t, best_loss1_v, best_loss2_v]