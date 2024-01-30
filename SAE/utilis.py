import torch
from plyfile import PlyData
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os as os

def initialize_weights_and_bias(m):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.0)


def init_weights(m):
    if isinstance(m, torch.nn.ParameterList):
        for param in m:
            torch.nn.init.xavier_uniform_(param.data)

    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
        initialize_weights_and_bias(m)


#from celi/antonias code
def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
          
#calculate mean face
def meanply(Data, typ = "Coma"):
    if os.path.isfile("mean_"+typ+".npy"):
        m = np.load("mean_"+typ+".npy")
    else:
        i=0
        if typ == "FWH":
            n = int(Data.shape[1]/3)
            for idx in range(len(Data)):
                vertex = Data[idx]
                verts_init = np.stack((vertex[:n], vertex[n:2*n], vertex[2*n:3*n]),axis=1)
                verts_init = verts_init.astype('float32')
        
                if i==0:
                    mean=verts_init
                    i=i+1
                else:
                    mean=mean+verts_init
        else:
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
        sc = 1000 if typ == "Coma" else 1    
        m = mean/len(Data)*sc
        np.save("mean_"+typ+".npy",m)
    return(m)



class EUCLoss(torch.nn.Module):
    def __init__(self):
        super(EUCLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction="none")
        loss = torch.mean(torch.sqrt(torch.sum(loss,axis =2)))
        return loss