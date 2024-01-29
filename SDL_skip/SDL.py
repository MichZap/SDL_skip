import torch
import torch.nn as nn
import math
from model_utilis import generate_inverse_series, IterativeSkip, MultiHeadAtt, activation_func



class SkipBlockD(nn.Module):
    def __init__(self,hidden_dim, conv_size, phi, eig_vals, prior_coef, activation='ReLU',dropout = 0, use_activation = 0, use_conv = True, use_eigenvals = False, use_norm = False, bias = True, reset =True, use_soft = False, use_att = False, aw = False, Skip = 2, n_heads = 0 ):
        super(SkipBlockD, self).__init__()
        
        d = 4 if use_eigenvals else 3
        
        if activation == "SwiGLU" and use_activation == -1:
           conv_size_in = int(conv_size/3)
           conv_size_out = conv_size_in*2
        else:
            conv_size_in = conv_size
            conv_size_out = conv_size
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        
        activation = activation_func(activation)
            
        self.use_att = use_att
        self.use_soft = use_soft
        self.n_heads = n_heads
           
        
        if use_att == False:
            self.l1 = nn.Linear(in_channels, hidden_dim ,bias)
            self.l2 = nn.Linear(hidden_dim, out_channels ,bias)
        else:
            if n_heads == 0:
                self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
                self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
                self.norm = lambda x: torch.nn.functional.normalize(x, p=1, dim=1)
            else:
                self.mh = MultiHeadAtt(hidden_dim,n_heads,dropout,use_norm,phi)

            
        self.l3 = nn.Linear(d, conv_size_out ,bias) if use_conv else nn.Identity()
        self.l4 = nn.Linear(conv_size_in, 3 ,bias) if use_conv else nn.Identity()
        self.activation1 = activation if use_activation >= 1 else nn.Identity()
        self.activation2 = activation if use_activation >= 2 else nn.Identity()
        self.activation3 = activation if use_activation == -1 else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(in_channels) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d) if use_norm else nn.Identity()
        self.soft = nn.Softmax(dim=2) if use_soft else nn.Identity()
        
        self.phi = phi
        self.eigv = eig_vals[:out_channels]
        self.use_eigenvals = use_eigenvals
        
        self.aw = aw
        
        if aw:
            self.weight = nn.Parameter(torch.Tensor(1, 1))
            nn.init.constant_(self.weight, 1)
        self.weights = nn.Parameter(torch.Tensor(2, 1))
        if reset:
            self.reset_parameters()
        nn.init.constant_(self.weights, prior_coef)
        
        self.Skip = Skip

            

    def forward(self, x):
        
        if self.Skip >= 1:
            z = torch.matmul(self.phi,x)
        
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.use_att:
            if self.n_heads == 0:
                score = torch.matmul(self.key,self.query)/torch.sqrt(torch.tensor(self.key.size(-1)))
                score = self.norm(score)
                y = torch.matmul(self.dropout(score).T,x)
            else:
                y = self.mh(x)
            
            
        else:
            
            y = self.l1(x.permute(0, 2, 1))
            y = self.activation1(y)
                
            y = self.l2(y)
            y = self.activation2(y)
                
            y = self.dropout(y).permute(0, 2, 1)
            if self.use_soft:
                y = self.soft(y/torch.sqrt(torch.tensor(y.size(2))))-0.5
        
        
        if self.Skip >= 2:
            y = (1-self.weights[0])*y+self.weights[0]*z

            
        if self.use_eigenvals:
            ev =torch.unsqueeze(self.eigv.repeat(x.size(0),1),2)
            y = torch.cat((y,ev),2)
            del ev
        y = self.norm2(y)
        y = self.l3(y)
        y = self.activation3(self.activation1(y))

        y = self.dropout(y)
        y = self.l4(y)
        y = self.activation2(y)
        y = self.dropout(y)
        

        if 1<= self.Skip <= 2:
            z = (1-self.weights[1])*y+self.weights[1]*z

        else:
            z = y
                
        if self.aw:
            z = self.weight*z
                
        
        del y
        torch.cuda.empty_cache()
        
        return z
    
    def reset_parameters(self):
        if self.use_att and self.n_heads ==0:
            torch.nn.init.xavier_uniform_(self.key, gain=1.0)
            torch.nn.init.xavier_uniform_(self.query, gain=1.0)
        if self.use_att == False:
            torch.nn.init.xavier_uniform_(self.l1.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
        if isinstance(self.l3, nn.Linear):
            torch.nn.init.xavier_uniform_(self.l3.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.l4.weight, gain=1.0)

class SkipBlockU(nn.Module):
    def __init__(self,hidden_dim, conv_size, phi, eig_vals, prior_coef, activation='ReLU',dropout = 0, use_activation = 0, use_conv = True, use_eigenvals = False, use_norm = False, bias = True, reset = True, use_soft= False,use_att = False, aw = False, Skip = 2, n_heads = 0):
        super(SkipBlockU, self).__init__()
        
        d = 4 if use_eigenvals else 3
        if activation == "SwiGLU" and use_activation == -1:
           conv_size_in = int(conv_size/3)
           conv_size_out = conv_size_in*2
        else:
            conv_size_in = conv_size
            conv_size_out = conv_size
            
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        
        activation = activation_func(activation)
            
        self.use_att = use_att
        self.use_soft = use_soft
        self.n_heads = n_heads
        
        if use_att == False:
            self.l1 = nn.Linear(in_channels, hidden_dim ,bias)
            self.l2 = nn.Linear(hidden_dim, out_channels ,bias)
        else:
            if n_heads == 0:
                self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
                self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
                self.norm = lambda x: torch.nn.functional.normalize(x, p=1, dim=1)
            else:
                self.mh = MultiHeadAtt(hidden_dim,n_heads,dropout,use_norm,phi)
        

            
        self.l3 = nn.Linear(d, conv_size_out ,bias) if use_conv else nn.Identity()
        self.l4 = nn.Linear(conv_size_in, 3 ,bias) if use_conv else nn.Identity()
        self.activation1 = activation if use_activation >= 1 else nn.Identity()
        self.activation2 = activation if use_activation == 2 else nn.Identity()
        self.activation3 = activation if use_activation == -1 else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(in_channels) if use_norm else nn.Identity()
        self.soft = nn.Softmax(dim=2) if use_soft else nn.Identity()
        
        self.phi = phi
        self.eigv = eig_vals[:in_channels]
        self.use_eigenvals = use_eigenvals
        self.aw = aw
        if aw:
            self.weight = nn.Parameter(torch.Tensor(1, 1))
            nn.init.constant_(self.weight, 1)
        self.weights = nn.Parameter(torch.Tensor(2, 1))
        if reset:
            self.reset_parameters()
        nn.init.constant_(self.weights, prior_coef)
        self.Skip = Skip
   
            

    def forward(self, x):
        
        y = x
        if self.use_eigenvals:
            ev = torch.unsqueeze(self.eigv.repeat(x.size(0),1),2)
            y = torch.cat((y,ev),2)
            del ev
        y = self.norm1(y)
        y = self.l3(y)
        y = self.activation3(self.activation1(y))

        y = self.dropout(y)
        y = self.l4(y)
        y = self.activation2(y)

        y = self.dropout(y)

        if self.Skip >= 2:
            y = (1-self.weights[1])*y+self.weights[1]*x

        
        y = self.norm2(y.permute(0, 2, 1)).permute(0, 2, 1)
        

        if self.use_att:
            if self.n_heads == 0:
                score = torch.matmul(self.key,self.query)/torch.sqrt(torch.tensor(self.key.size(-1)))
                score = self.norm(score)
                y = torch.matmul(self.dropout(score).T,y)
            else:
                y = self.mh(x)
        else:
            y = self.l1(y.permute(0, 2, 1))
            y = self.activation1(y)

            y = self.l2(y)
            y = self.activation2(y)
            y = self.dropout(y).permute(0, 2, 1)
            if self.use_soft:
                y = self.soft(y/torch.sqrt(torch.tensor(y.size(2))))-0.5
                
        if 1<= self.Skip <= 2:
            z = torch.matmul(self.phi,x)                
            y = (1-self.weights[0])*y+self.weights[0]*z

            del z
                
        if self.aw:
            y = self.weight*y
        
        torch.cuda.empty_cache()
        
        return y
    
    def reset_parameters(self):
        if self.use_att and self.n_heads ==0:
            torch.nn.init.xavier_uniform_(self.key, gain=1.0)
            torch.nn.init.xavier_uniform_(self.query, gain=1.0)
        if self.use_att == False:
            torch.nn.init.xavier_uniform_(self.l1.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
        if isinstance(self.l3, nn.Linear):
            torch.nn.init.xavier_uniform_(self.l3.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.l4.weight, gain=1.0)


class SDL_skip(nn.Module):
    def __init__(self, opt,eig_vecs,eig_vals):
        super().__init__()
        
        k = len(opt["nb_freq"])
        
        #n_w = 4 + 5(2**k-2) # 2 per skipping block + 1 per additional skip sum_{i=1}_{k}2**(i+1) + sum_{i=2}_{k}2**(i-1) use formula for power 2 series
        phi = [torch.tensor(eig_vecs[:,:opt["nb_freq"][i]],device = opt["device"],requires_grad=False) for i in range(k)]
                            
        Spectral_M_down = []
        l = k
        
        while l>0:
            Spectral_M_down.append(phi[l-1].t())
            j = l
            while l < k:
                i = l        
                l = l + j
                M = torch.matmul(phi[l-1].t(),phi[i-1])
                Spectral_M_down.append(M)
                del M
            l = int(math.floor(j/2))
        

        inversindex = generate_inverse_series(len(Spectral_M_down)) 
        del phi
        torch.cuda.empty_cache()
        
        eig_vals = torch.tensor(eig_vals,device = opt["device"],requires_grad=False)                 
        
        self.SkipBlocksDown = nn.ModuleList(
            [
            SkipBlockD(
                hidden_dim = opt["hidden_dim"][i],
                conv_size = opt["conv_size"][i],
                phi = Spectral_M_down[i],
                eig_vals = eig_vals,
                prior_coef = opt["prior_coef"],
                activation = opt["activation"],
                dropout = opt["dropout"] if 'dropout' in opt else 0,
                use_activation = opt["use_activation"],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_soft = opt["use_soft"] if "use_soft" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                aw = opt["aw"] if "aw" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                n_heads = opt["n_heads"] if "n_heads" in opt else 0,
                
            )
            for i in range(len(Spectral_M_down))]
        )
        
        self.SkipBlocksUp = nn.ModuleList(
            [
            SkipBlockU(
                hidden_dim = opt["hidden_dim"][inversindex[i]],
                conv_size = opt["conv_size"][inversindex[i]],
                phi = Spectral_M_down[inversindex[i]].t(),
                eig_vals = eig_vals,
                prior_coef = opt["prior_coef"],
                activation = opt["activation"],
                dropout = opt["dropout"] if 'dropout' in opt else 0,
                use_activation = opt["use_activation"],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_soft = opt["use_soft"] if "use_soft" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                aw = opt["aw"] if "aw" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                n_heads = opt["n_heads"] if "n_heads" in opt else 0,
            )
            for i in range(len(Spectral_M_down))]
        )
        
        del Spectral_M_down
        torch.cuda.empty_cache()


        
        
    def forward(self, x):
        
        x = IterativeSkip(x,self.SkipBlocksDown)
        x = IterativeSkip(x,self.SkipBlocksUp)

        return x