from utilis import EUCLoss, gnn_model_summary
import torch
import numpy as np
from SDL import SDL_skip
from SDL_simple import SDL_skip_simple
from Dataloader import load_data
from torch.utils.data import DataLoader, ConcatDataset
from linear import linear
import math
import time
import os
from glob import glob
from utilis import Mesh

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 1000000

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
    #outfile_vec = config["path_decomp"] +"\\"+ str(f) +"eig_vecs.npy"
    #outfile_val = config["path_decomp"] +"\\"+ str(f) +"eig_vals.npy"
    outfile_vec = os.path.join(config["path_decomp"],str(f)+"eig_vecs.npy")
    outfile_val = os.path.join(config["path_decomp"],str(f)+"eig_vals.npy")

    eig_vecs = np.load(outfile_vec)
    eig_vals = np.load(outfile_val)
    folder_path = config["folder"]
    file_path = next(iter(glob(f"{folder_path}/*.ply")), None)
    Template_mesh = Mesh(filename = file_path)
    Adj = Template_mesh.Adj
    freq = min(config["nb_freq"])

    phi = torch.tensor(eig_vecs[:,:freq], dtype=torch.float32).cuda()

    simple = config["simple"] if 'simple' in config else False
    if simple == True:
        model = SDL_skip_simple(config,eig_vecs,eig_vals,Adj)
    elif simple == "linear":
        model =linear(config,eig_vecs,Adj)
    else:
        model = SDL_skip(config,eig_vecs,eig_vals)
        
    model.to(config["device"])
    if config["print"] == True:
        gnn_model_summary(model)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = config["shed_fac"],patience=config["shed_pat"],min_lr = config["learning_rate"]/100)
    early_stopper = EarlyStopper(patience=config["es_pat"], min_delta=config["min_delta"])
    criterion = torch.nn.L1Loss()
    criterion2 = EUCLoss()
    
    
    
    train_set, val_set = load_data(config["p"],config["deg"],config["root_dir"],config["folder"],config["mat_path"],[0,1,2], config["typ"],config["sym"],config["sd"],config["SSH"],config["p_asym"],config["k"],config["flip"],config["rot"],config["sigma"],config["aug_mean"],config["comp"],eig_vecs,config["unique"])

    if config["add_data"] == True:
        _, train_set2 = load_data(0,config["root_dir"],config["folder_add"],[0,1,2], "BBF_add")
        train_set = ConcatDataset([train_set,train_set2])
        
    trainloader = DataLoader(train_set,batch_size=config["train_batch_size"],shuffle=True,num_workers=config["num_workers"])
    valloader = DataLoader(val_set,batch_size=config["test_batch_size"],num_workers=config["num_workers"])
    
    best_epoch = 0
    min_loss2_t = 9999
    best_loss1_t = 9999
    best_loss2_t = 9999
    best_loss1_v = 9999
    best_loss2_v = 9999
    
    config["sd"] = config["sd"]  if "sd" in config else False
    if config["sd"] == True:
        sd = np.load("sd_"+config["typ"]+".npy")
    
    if config["SSH"] == True:
        SSH = config["SSH"]

        sym = config["sym"]
        if sym != False:
            sym = True
        typ = config["typ"]
        
        mean = np.load(file = f"mean_sym_{int(sym)}_SSH_{int(SSH)}_{typ}.npy")
        mean = torch.tensor(mean,dtype=torch.float32).to('cuda')
        

        
    

    
    #train
    for epoch in range(1, config["epochs"]):
        #logs = {}
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        running_loss2 = 0.0 
        running_loss3 = 0.0 
        
        for data in trainloader:  # Iterate in batches over the training dataset.
             pts = data['points'].to('cuda')
             if simple == True:
                 out = model(pts)
                 l2 = 0

             else:
                 out, outs = model(pts)
                 if config["lambda"] > 0:
                     l2 = criterion(outs, torch.matmul(phi.T,pts))
                 else:
                     l2 = 0

            
             if config["sd"]:
                sdt = torch.tensor(sd).unsqueeze(0).expand(len(data["points"]),f,3).to('cuda')
                loss = criterion(out*sdt, pts*sdt) 
                loss2 = criterion2(out*sdt, pts*sdt)
             else:
               if 'SSH' in data:
                   SSH = data['SSH'].unsqueeze(-1).to('cuda')
                   loss = criterion((out+mean)*SSH, (pts+mean)*SSH)
                   loss2 = criterion2((out+mean)*SSH, (pts+mean)*SSH)
               else:
                   loss = criterion(out, pts) 
                   loss2 = criterion2(out, pts)
             
             loss = loss + config["lambda"]*l2
             
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             for param in model.parameters():
                param.grad = None  # Clear gradients.
             
            
             running_loss += loss.item()*len(data["points"])
             running_loss2 += loss2.item()*len(data["points"])
             running_loss3 += l2.item()*len(data["points"]) if simple == False and config["lambda"] > 0 else 0
             del out, loss, loss2
             torch.cuda.empty_cache()
            
        epoch_loss = running_loss/len(trainloader.dataset)
        epoch_loss2 = running_loss2/len(trainloader.dataset)
        epoch_loss3 = running_loss3/len(trainloader.dataset)
        l1train  = epoch_loss
        l2train  = epoch_loss2
        l3train  = epoch_loss3
        
        epoch_time = time.time() - start_time
        
        #start_time = time.time()
        torch.cuda.empty_cache()
        model.eval()
        
        running_loss = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0 
        
        with torch.no_grad():
            for data in valloader:  # Iterate in batches over the training dataset.
                 pts = data["points"].to("cuda")
                 if simple == True:
                     out = model(pts)
                     l2 = 0
                 else:
                     out, outs = model(pts)
                     if config["lambda"] > 0:
                         l2 = criterion(outs, torch.matmul(phi.T,pts))
                     else:
                         l2 = 0

                 
                 if config["sd"]:
                     sdt = torch.tensor(sd).unsqueeze(0).expand(len(data["points"]),f,3).to('cuda')
                     loss = criterion(out*sdt, pts*sdt) 
                     loss2 = criterion2(out*sdt, pts*sdt)
                 else:
                    if 'SSH' in data:
                        SSH = data['SSH'].unsqueeze(-1).to('cuda')
                        loss = criterion((out+mean)*SSH, (pts+mean)*SSH)
                        loss2 = criterion2((out+mean)*SSH, (pts+mean)*SSH)

                    else:
                        loss = criterion(out, pts) 
                        loss2 = criterion2(out, pts)
                        
                 loss = loss + config["lambda"]*l2
                 
                 running_loss += loss.item()*len(data["points"])
                 running_loss2 += loss2.item()*len(data["points"])
                 running_loss3 += l2.item()*len(data["points"]) if simple == False and config["lambda"] > 0 else 0
                 
                 del out, loss, loss2
                 torch.cuda.empty_cache()
           
        epoch_loss = running_loss/len(valloader.dataset)
        epoch_loss2 = running_loss2/len(valloader.dataset)
        epoch_loss3 = running_loss3/len(valloader.dataset)
        
        if min_loss2_t  > l2train:
            min_loss2_t = l2train
            
        scheduler.step(epoch_loss)
        if early_stopper.early_stop(epoch_loss):             
            break
        
        if epoch%config["print_epoch"]==0 or epoch == 1:
           print("EP:",f"{epoch:>5}",
                 " MAE_v:",f"{np.round(epoch_loss,2):.2f}",
                 " MAE_sv:", f"{np.round(epoch_loss3,2):.2f}",
                 " EUC_v:", f"{np.round(epoch_loss2,2):.2f}",              
                 " MAE_t:", f"{np.round(l1train,2):.2f}",
                 " MAE_st:", f"{np.round(l3train,2):.2f}",
                 " EUC_t:", f"{np.round(l2train,2):.2f}",
                 " EP time:", f"{np.round(epoch_time,2):.2f}"
                 )
           if simple == True and config["gate"]==False:
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
