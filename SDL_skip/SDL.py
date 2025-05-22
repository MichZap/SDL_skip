import torch
import torch.nn as nn
import math
import numpy as np
from model_utilis import generate_inverse_series, IterativeSkip, MultiHeadAtt, activation_func, Att, positional_encoding, DyT



class SkipBlockD(nn.Module):
    def __init__(self,hidden_dim, conv_size, phi, eig_vals, prior_coef, activation='ReLU',dropout = 0, use_activation = 0,
                 use_conv = True, use_eigenvals = False, use_norm = False, bias = True, reset =True, use_soft = False,
                 use_att = False, use_norm_att = True, aw = False, Skip = 2, n_heads = 0, value = True, learn = True, 
                 sqrt = True , wdim = False, vinit = False, eb_dim = None, init_a = 0):
        super(SkipBlockD, self).__init__()
        
        self.learn = learn
        
        
        d = 4 if use_eigenvals else 3
        
        if activation == "SwiGLU" and use_activation == -1:
           conv_size_in = int(conv_size/3)
           conv_size_out = conv_size_in*2
        else:
            conv_size_in = conv_size
            conv_size_out = conv_size
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        max_channels = max(in_channels,out_channels)
        
        activation = activation_func(activation)
            
        self.use_att = use_att
        self.use_soft = use_soft
        self.n_heads = n_heads 
        self.phi = phi
        self.eigv = eig_vals[:out_channels]
        self.use_eigenvals = use_eigenvals
        self.aw = aw
        self.Skip = Skip
        self.sqrt = sqrt
        self.vinit = vinit
           
        if learn:
            device = phi.device 
            if use_att == False:
                self.l1 = nn.Linear(in_channels, hidden_dim ,bias)
                self.l2 = nn.Linear(hidden_dim, out_channels ,bias)
            else:
                
                if n_heads == 0 and vinit == False:
                    self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
                    self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
                    self.norm = (lambda x: torch.nn.functional.normalize(x, p=1, dim=1)) if use_norm_att else nn.Identity() 
                elif vinit == True:
                    self.pos = positional_encoding(max_channels, 3).to(device)
                    self.att = Att(hidden_dim,3,3,dropout,use_norm_att,value,sqrt,use_att,False,False,reset,vinit,eb_dim)
                else:
                    self.mh = MultiHeadAtt(hidden_dim,n_heads,dropout,use_norm_att,phi,value,sqrt)
    
                
            self.l3 = nn.Linear(d, conv_size_out ,bias) if use_conv else nn.Identity()
            self.l4 = nn.Linear(conv_size_in, 3 ,bias) if use_conv else nn.Identity()
            self.activation1 = activation if use_activation in [1,2] else nn.Identity()
            self.activation2 = activation if use_activation == 2 else nn.Identity()
            self.activation3 = activation if use_activation == -1 else nn.Identity()
            self.dropout = nn.Dropout(p=dropout)
            
            if use_norm == True:
                if init_a == 0:
                    self.norm1 = nn.LayerNorm(3)
                    self.norm2 = nn.LayerNorm(d)
                else:
                    init_a = torch.tensor(init_a).to(device)
                    self.norm1 = DyT(3,init_a)
                    self.norm2 = DyT(d,init_a) 
            else:
                self.norm1 = nn.Identity()
                self.norm2 = nn.Identity()
                
            self.soft = nn.Softmax(dim=2) if use_soft else nn.Identity()

            if wdim == True:
                self.weights_0 = nn.Parameter(torch.Tensor(1,out_channels,1))
                self.weights_1 = nn.Parameter(torch.Tensor(1,out_channels,1))
            else:
                self.weights_0 = nn.Parameter(torch.Tensor(1))
                self.weights_1 = nn.Parameter(torch.Tensor(1))
            
            if reset:
                self.reset_parameters()
            nn.init.constant_(self.weights_0, prior_coef)
            nn.init.constant_(self.weights_1, prior_coef)
        

        
        if aw:
            self.weight = nn.Parameter(torch.Tensor(1, 1))
            nn.init.constant_(self.weight, 1)
            

        

            

    def forward(self, x):
        
        if self.learn == False:
            return self.weight * torch.matmul(self.phi,x)
        
        if self.Skip >= 1:
            z = torch.matmul(self.phi,x)
        
        x = self.norm1(x)
        
        if self.use_att:
            if self.n_heads == 0 and self.vinit == False:
                n = torch.sqrt(torch.tensor(self.key.size(-1))) if self.sqrt else 1
                score = torch.matmul(self.key,self.query)/n
                score = self.norm(score)
                y = torch.matmul(self.dropout(score).T,x)
            elif self.vinit == True:
                x_enc = torch.cat([x, self.pos[:x.size(1),:].unsqueeze(0).expand(x.size(0), -1, -1)], dim=-1)
                z_enc = torch.cat([z, self.pos[:z.size(1),:].unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
                y = self.att(x_enc,z_enc)
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
            y = (1-self.weights_0)*y+self.weights_0*z

            
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
            z = (1-self.weights_1)*y + self.weights_1*z

        else:
            z = y
                
        if self.aw:
            z = self.weight*z
                
        
        del y
        torch.cuda.empty_cache()
        
        return z, 0
    
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
    def __init__(self,hidden_dim, conv_size, phi, eig_vals, prior_coef, activation='ReLU',dropout = 0, use_activation = 0,
                 use_conv = True, use_eigenvals = False, use_norm = False, bias = True, reset = True, use_soft= False,
                 use_att = False, use_norm_att = True, aw = False, Skip = 2, n_heads = 0, value = True, learn = True, 
                 sqrt = True , FFW = False, wdim = False, vinit = False, eb_dim = None, init_a = 0):
        super(SkipBlockU, self).__init__()
        
        self.learn = learn
        
        d = 4 if use_eigenvals else 3
        if activation == "SwiGLU" and use_activation == -1:
           conv_size_in = int(conv_size/3)
           conv_size_out = conv_size_in*2
        else:
            conv_size_in = conv_size
            conv_size_out = conv_size
            
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        max_channels = max(in_channels,out_channels)
        
        activation = activation_func(activation)
            
        self.use_att = use_att
        self.use_soft = use_soft
        self.n_heads = n_heads
        self.phi = phi
        self.eigv = eig_vals[:in_channels]
        self.use_eigenvals = use_eigenvals
        self.aw = aw
        self.Skip = Skip
        self.sqrt = sqrt
        self.vinit = vinit
        
        if learn:
            
            device = phi.device 
            
            if use_att == False:
                self.l1 = nn.Linear(in_channels, hidden_dim ,bias)
                self.l2 = nn.Linear(hidden_dim, out_channels ,bias)
            else:
                if n_heads == 0 and vinit == False:
                    self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
                    self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
                    self.norm = (lambda x: torch.nn.functional.normalize(x, p=1, dim=1)) if use_norm_att else nn.Identity() 
                elif vinit == True:
                    self.pos = positional_encoding(max_channels, 3).to(device)
                    self.att = Att(hidden_dim,3,3,dropout,use_norm_att,value,sqrt,use_att,False,False,reset,vinit,eb_dim)
                else:
                    self.mh = MultiHeadAtt(hidden_dim,n_heads,dropout,use_norm_att,phi,value,sqrt)                    
            
    
                
            self.l3 = nn.Linear(d, conv_size_out ,bias) if use_conv else nn.Identity()
            self.l4 = nn.Linear(conv_size_in, 3 ,bias) if use_conv else nn.Identity()
            self.activation1 = activation if use_activation in [1,2] else nn.Identity()
            self.activation2 = activation if use_activation == 2 else nn.Identity()
            self.activation3 = activation if use_activation == -1 else nn.Identity()
            self.dropout = nn.Dropout(p=dropout)
            
            if use_norm == True:
                if init_a == 0:
                    self.norm1 = nn.LayerNorm(d)
                    self.norm2 = nn.LayerNorm(3)
                else:
                    init_a = torch.tensor(init_a).to(device)
                    self.norm1 = DyT(d,init_a)
                    self.norm2 = DyT(3,init_a) 
            else:
                self.norm1 = nn.Identity()
                self.norm2 = nn.Identity()

            self.soft = nn.Softmax(dim=2) if use_soft else nn.Identity()
            self.weights_1 = nn.Parameter(torch.Tensor(1))
            if wdim == True:
                self.weights_0 = nn.Parameter(torch.Tensor(1,out_channels,1))    
            else:
                self.weights_0 = nn.Parameter(torch.Tensor(1))
            

            self.FFW = FeedForward(in_channels*3, conv_size, in_channels*3, activation, use_activation) if FFW == True else nn.Identity()
            self.use_FFW = FFW
            
            if reset:
                self.reset_parameters()
            nn.init.constant_(self.weights_0, prior_coef)
            nn.init.constant_(self.weights_1, prior_coef)
        

        if aw:
            self.weight = nn.Parameter(torch.Tensor(1, 1))
            nn.init.constant_(self.weight, 1)
        
   
            

    def forward(self, x):
        
        if self.learn == False:
            return self.weight * torch.matmul(self.phi,x)
        
        xs = self.FFW(torch.flatten(x,start_dim = 1)).view(x.shape)
        
        y = x - xs if self.use_FFW == True else x
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
            y = (1-self.weights_1)*y+self.weights_1*x

        
        y = self.norm2(y)
        

        if self.use_att:
            if self.n_heads == 0 and self.vinit == False:
                n = torch.sqrt(torch.tensor(self.key.size(-1))) if self.sqrt else 1
                score = torch.matmul(self.key,self.query)/n
                score = self.norm(score)
                y = torch.matmul(self.dropout(score).T,y)
            elif self.vinit == True:
                x_p = torch.matmul(self.phi,x)
                x_enc = torch.cat([x_p, self.pos[:x_p.size(1),:].unsqueeze(0).expand(x_p.size(0), -1, -1)], dim=-1)
                y_enc = torch.cat([y, self.pos[:y.size(1),:].unsqueeze(0).expand(y.size(0), -1, -1)], dim=-1)
                y = self.att(y_enc,x_enc)
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
            z = torch.matmul(self.phi,xs)            
            y = y*(1-self.weights_0) + z*self.weights_0

            del z
                
        if self.aw:
            y = self.weight*y
        
        torch.cuda.empty_cache()
        
        if self.use_FFW == False:
            xs = 0
        
        return y, xs

    
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
            
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='ReLU', use_activation = 0):
        super(FeedForward, self).__init__()
        
        if hidden_dim*input_dim>5e5:
            hidden_dim = int(5e5/input_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation if use_activation != 0 else nn.Identity()
    
    def forward(self, y):
        y = self.fc2(self.activation(self.fc1(y))) + y
        return y


class SDL_skip(nn.Module):
    def __init__(self, opt,eig_vecs,eig_vals):
        super().__init__()
        
        k = len(opt["nb_freq"])
        
        #n_w = 4 + 5(2**k-2) # 2 per skipping block + 1 per additional skip sum_{i=1}_{k}2**(i+1) + sum_{i=2}_{k}2**(i-1) use formula for power 2 series
        phi = [torch.tensor(eig_vecs[:,:opt["nb_freq"][i]],device = opt["device"],requires_grad=False, dtype=torch.float32) for i in range(k)]
                            
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
        
        eig_vals = torch.tensor(eig_vals,device = opt["device"],requires_grad=False, dtype=torch.float32)

        learn = opt["learn"] if "learn" in opt else True
        n = sum([2**i for i in range(opt["depth"]+1)])
        
        if learn == False:
            learn = np.repeat([False],n)
            learn[-2**opt["depth"]:] = True
        else:
            learn = np.repeat([True],n)
        
        FFW = np.repeat([False],n)    
        if opt["FFW"] == True:
            FFW[0] = True
        
                 
        
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
                value = opt["value"] if "value" in opt else True,
                learn = learn[i],
                use_norm_att = opt["use_norm_att"] if "use_norm_att" in opt else True,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                wdim = opt["wdim"] if "sqrt" in opt else False,
                vinit = opt["vinit"] if "sqrt" in opt else False,
                eb_dim = opt["eb_dim"] if "eb_dim" in opt else None,
                init_a = opt["init_a"] if "init_a" in opt else 0,
                
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
                value = opt["value"] if "value" in opt else True,
                learn = learn[inversindex[i]],
                use_norm_att = opt["use_norm_att"] if "use_norm_att" in opt else True,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                FFW = FFW[i],
                wdim = opt["wdim"] if "sqrt" in opt else False,
                vinit = opt["vinit"] if "sqrt" in opt else False,
                eb_dim = opt["eb_dim"] if "eb_dim" in opt else None,
                init_a = opt["init_a"] if "init_a" in opt else 0,
            )
            for i in range(len(Spectral_M_down))]
        )
        
        del Spectral_M_down
        torch.cuda.empty_cache()

    def encoder(self,x):
        z,_ = IterativeSkip(x,self.SkipBlocksDown)
        return z
    
    def decoder(self,z):
        x,xs = IterativeSkip(z,self.SkipBlocksUp)
        return x,xs

        
    def forward(self, x):
        
        z = self.encoder(x)
        x = self.decoder(z)

        return x
    
class Spectral_decomp(nn.Module):
    def __init__(self, opt,eig_vecs):
        super().__init__()
        

        self.phi = torch.tensor(eig_vecs[:,:opt["nb_freq"][-1]],device = opt["device"],requires_grad=False, dtype=torch.float32)
                            


    def encoder(self,x):
        z = torch.matmul(self.phi.T,x)
        return z
    
    def decoder(self,z):
        x = torch.matmul(self.phi,z)
        return x

        
    def forward(self, x):
        
        z = self.encoder(x)
        x = self.decoder(z)

        return x,z