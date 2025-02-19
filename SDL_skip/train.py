from utilis import EUCLoss, gnn_model_summary
import torch
import numpy as np
from SDL import SDL_skip
from SDL_simple import SDL_skip_simple
from Dataloader import load_data
from torch.utils.data import DataLoader, ConcatDataset
import math
import time



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 100

    def early_stop(self, validation_loss):
        if math.isnan(validation_loss):
           self.counter += 1
           if self.counter >= self.patience:
               return True
        else:
            if validation_loss < (self.min_validation_loss - self.min_delta):
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
    

    
def train_bf(config):
    print("________________________________START___________________________________")
    if config["use_conv"]==False:
        config["use_eigenvals"]=False
    #liveloss = PlotLosses()
    #setup Model and Optimizer
    #setup Model and Optimizer
    if config["typ"] == "Coma":
        f = "ComaData_5023"
    elif config["typ"] == "FWH":
        f = 11510
    else:
        f = 31027
        
    
    #load eigenvectors and value
    outfile_vec = config["path_decomp"] +"\\"+ str(f) +"eig_vecs.npy"
    outfile_val = config["path_decomp"] +"\\"+ str(f) +"eig_vals.npy"

    eig_vecs = np.load(outfile_vec)
    eig_vals = np.load(outfile_val)

    freq = max(config["nb_freq"])

    eig_vals = eig_vals[:freq]
    eig_vecs = eig_vecs[:,:freq]

    simple = config["simple"] if 'simple' in config else False
    if simple:
        model = SDL_skip_simple(config,eig_vecs,eig_vals)
    else:
        model = SDL_skip(config,eig_vecs,eig_vals)
    
    gnn_model_summary(model)
    
    model.to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = config["shed_fac"],patience=config["shed_pat"],min_lr = config["learning_rate"]/100)
    early_stopper = EarlyStopper(patience=config["es_pat"], min_delta=config["min_delta"])
    criterion = torch.nn.L1Loss()
    criterion2 = EUCLoss()
    
    
    
    train_set, val_set = load_data(config["p"],config["root_dir"],config["folder"],config["mat_path"],[0,1,2], config["typ"],config["sym"],config["sd"])
    
    if config["add_data"] == True:
        _, train_set2 = load_data(0,config["root_dir"],config["folder_add"],[0,1,2], "BBF_add")
        train_set = ConcatDataset([train_set,train_set2])
        
    trainloader = DataLoader(train_set,batch_size=config["train_batch_size"],shuffle=True)
    valloader = DataLoader(val_set,batch_size=config["test_batch_size"])
    
    best_epoch = 0
    min_loss2_t = 9999
    best_loss1_t = 9999
    best_loss2_t = 9999
    best_loss1_v = 9999
    best_loss2_v = 9999
    
    config["sd"] = config["sd"]  if "sd" in config else False
    if config["sd"]:
        sd = np.load("sd_"+config["typ"]+".npy")

    
    #train
    for epoch in range(1, config["epochs"]):
        #logs = {}
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        running_loss2 = 0.0 

        
        for data in trainloader:  # Iterate in batches over the training dataset.
             out = model(data['points'].to('cuda'))  # Perform a single forward pass.
             loss = criterion(out, data['points'].cuda())  # Compute the loss.
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             for param in model.parameters():
                param.grad = None  # Clear gradients.
             
             if config["sd"]:
                 sdt = torch.tensor(sd).unsqueeze(0).expand(len(data["points"]),f,3).to('cuda')
                 loss2 = criterion2(out*sdt, data['points'].cuda()*sdt)
             else:
                 loss2 = criterion2(out, data['points'].cuda())

             running_loss += loss.item()*len(data["points"])
             running_loss2 += loss2.item()*len(data["points"])
             del out, loss, loss2
             torch.cuda.empty_cache()
            
        epoch_loss = running_loss/len(trainloader.dataset)
        epoch_loss2 = running_loss2/len(trainloader.dataset)
        l1train  = epoch_loss
        l2train  = epoch_loss2
        #prefix=""

        epoch_time = time.time() - start_time
        
        #start_time = time.time()
        torch.cuda.empty_cache()
        model.eval()
        
        running_loss = 0.0
        running_loss2 = 0.0 
        
        with torch.no_grad():
            for data in valloader:  # Iterate in batches over the training dataset.
                 out = model(data['points'].to('cuda'))  # Perform a single forward pass.
                 loss = criterion(out, data['points'].cuda())  # Compute the loss.
                 
                 if config["sd"]:
                     sdt = torch.tensor(sd).unsqueeze(0).expand(len(data["points"]),f,3).to('cuda')
                     loss2 = criterion2(out*sdt, data['points'].cuda()*sdt)
                 else:
                     loss2 = criterion2(out, data['points'].cuda())


                 running_loss += loss.item()*len(data["points"])
                 running_loss2 += loss2.item()*len(data["points"])
                 del out, loss, loss2
                 torch.cuda.empty_cache()
           
        epoch_loss = running_loss/len(valloader.dataset)
        epoch_loss2 = running_loss2/len(valloader.dataset)
        
        
        if min_loss2_t  > l2train:
            min_loss2_t = l2train
            
        scheduler.step(epoch_loss)
        if early_stopper.early_stop(epoch_loss):             
            break
        
        
        if epoch%config["print_epoch"]==0 or epoch == 1:
           print("epoch:",epoch,
                 "MAE_loss_val:",np.round(epoch_loss,2),
                 "EUC_loss_val:", np.round(epoch_loss2,2),
                 "MAE_loss_train:", np.round(l1train,2),
                 "EUC_loss_train:", np.round(l2train,2),
                 "Epoch time:", np.round(epoch_time,2)
                 )
           if simple and config["gate"]==False:
               print(np.round(model.weightsD.cpu().detach().numpy(),2))
               print(np.round(model.weightsU.cpu().detach().numpy(),2))
        
        if epoch_loss < best_loss1_t:
            best_epoch = epoch
            best_loss1_t = epoch_loss
            best_loss2_t = epoch_loss2
            best_loss1_v = l1train
            best_loss2_v = l2train
            
        if epoch == config["save_epoch"] and epoch > 0:
            PATH = str(config["nb_freq"][-1])+"_SDL_skip.pt"
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)        
    print("Finished Training")
    return [config["nb_freq"][-1],epoch, best_epoch, best_loss1_t, best_loss2_t, best_loss1_v, best_loss2_v,min_loss2_t]
