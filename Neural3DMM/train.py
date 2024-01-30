from utils import EUCLoss
import torch
import numpy as np
import pickle
import os
from models import SpiralAutoencoder
from utils import Mesh
from Dataloader import load_data, autoencoder_dataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from spiral_utils import get_adj_trigs, generate_spirals


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
    print("Set up model:")

    #load precalculated Down- and Upsampling matrices, created in 3.6 env:
    with open(os.path.join(config["folder"],'downsampling_matrices.pkl'), 'rb') as fp:
        downsampling_matrices = pickle.load(fp)
            
        M_verts_faces = downsampling_matrices['M_verts_faces']
    
        M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
    
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']
    
    
    for i in range(len(M_verts_faces)):
        M[i].v=np.vstack((M[i].v, [0,0,0]))
        
    nV_ref = []
    ref_mean = np.mean(M[0].v, axis=0)
    for i in range(len(M)):
        nv = 0.1 * (M[i].v - ref_mean)
        nV_ref.append(nv)
    
    for i in range(len(M_verts_faces)):
        M[i].v=M[i].v[:-1]
        
    tV_ref = [torch.from_numpy(s).float().to(config["device"]) for s in nV_ref]
    
    reference_points = config["reference_points"].copy()
    for i in range(len([4,4,4,4])):
        dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
        reference_points.append(np.argmin(dist,axis=0).tolist())
        
    sizes = [x.v.shape[0] for x in M]
    
    template_file_path = config["template_file_path"]
    template_mesh = Mesh(filename=template_file_path)
    
    Adj, Trigs = get_adj_trigs(A, F, template_mesh, meshpackage = 'mpi-mesh')
    
    spirals_np, spiral_sizes,spirals = generate_spirals(config["step_sizes"], 
                                                    M, Adj, Trigs, 
                                                    reference_points = reference_points, 
                                                    dilation = config["dilation"], random = False, 
                                                    meshpackage = 'mpi-mesh', 
                                                    counter_clockwise = True)
    
    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((D[i].shape[0]+1,D[i].shape[1]+1))
        u = np.zeros((U[i].shape[0]+1,U[i].shape[1]+1))
        d[:-1,:-1] = D[i].todense()
        u[:-1,:-1] = U[i].todense()
        d[-1,-1] = 1
        u[-1,-1] = 1
        bD.append(d)
        bU.append(u)
        
    
    tspirals = [torch.from_numpy(s).long().to(config["device"]) for s in spirals_np]
    tD = [torch.from_numpy(s).float().to(config["device"]) for s in bD]
    tU = [torch.from_numpy(s).float().to(config["device"]) for s in bU]
    
    #setup Model and Optimizer    
    model = SpiralAutoencoder(
                          sizes=sizes,
                          spiral_sizes=spiral_sizes,
                          spirals=tspirals,
                          D=tD, 
                          U=tU,
                          V_ref=tV_ref,
                          config=config
                         )
    model.to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = config["shed_fac"],patience=config["shed_pat"],min_lr = config["learning_rate"]/100)
    early_stopper = EarlyStopper(patience=config["es_pat"], min_delta=0)
    criterion = torch.nn.L1Loss()
    criterion2 = EUCLoss()
    
    
    print("Load data:")
    train_set, val_set = load_data(config["p"],config["root_dir"],config["folder"],[0,1,2],config["typ"])

    trainloader = DataLoader(train_set,batch_size=config["train_batch_size"],shuffle=True)
    valloader = DataLoader(val_set,batch_size=config["test_batch_size"])
    print("________________________________START___________________________________")
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

        for data in trainloader:  # Iterate in batches over the training dataset.
             out = model(data['points'].to('cuda'))  # Perform a single forward pass.
             loss = criterion(out, data['points'].cuda())  # Compute the loss.
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             #print(torch.cuda.memory_allocated()/1024**2," MB")
             for param in model.parameters():
                param.grad = None  # Clear gradients.
                
             loss2 = criterion2(out, data['points'].cuda())

             running_loss += loss.item()*len(data["points"])
             running_loss2 += loss2.item()*len(data["points"])

             torch.cuda.empty_cache()
            
        epoch_loss = running_loss/len(trainloader.dataset)
        epoch_loss2 = running_loss2/len(trainloader.dataset)
        l1train  = epoch_loss
        l2train  = epoch_loss2
        torch.cuda.empty_cache()
        
        model.eval()
        
        running_loss = 0.0
        running_loss2 = 0.0
        
        with torch.no_grad():
            for data in valloader:  # Iterate in batches over the training dataset.
                 out = model(data['points'].to('cuda'))  # Perform a single forward pass.
                 loss = criterion(out, data['points'].cuda())  # Compute the loss.
                 loss2 = criterion2(out, data['points'].cuda())                   

                 running_loss += loss.item()*len(data["points"])
                 running_loss2 += loss2.item()*len(data["points"])
                 torch.cuda.empty_cache()
           
        epoch_loss = running_loss/len(valloader.dataset)
        epoch_loss2 = running_loss2/len(valloader.dataset)
        
        
        scheduler.step(epoch_loss)
        if early_stopper.early_stop(epoch_loss):
                    
            break
        
        if epoch%config["print_epoch"]==0:
            print("epoch:",epoch,"MAE_loss_val:", epoch_loss, "EUC_loss_val:", epoch_loss2,"MAE_loss_train:", l1train, "EUC_loss_train:", l2train)
        if epoch_loss < best_loss1_t:
            best_epoch = epoch
            best_loss1_t = epoch_loss
            best_loss2_t = epoch_loss2
            best_loss1_v = l1train
            best_loss2_v = l2train
    
    PATH = str(config["size_latent"])+"_N3DMM_Att.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
    print("Finished Training")
    
    return [config["size_latent"],epoch, best_epoch, best_loss1_t, best_loss2_t, best_loss1_v, best_loss2_v]