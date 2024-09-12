import torch
import torch.nn as nn
from model_utilis import  activation_func, MultiHeadAtt



#class for final weighted skip connection to the latent space with num learnable weights = num spectral freq   
class Skip(nn.Module):
    def __init__(
            self,
            phi,
            up = False,
            ):
        super(Skip, self).__init__()
        
        self.phi = phi
        self.up = up
        
    def forward(self, x, z, weights):        
        
        phi = self.phi if self.up else self.phi.T
        
        if weights.dim()==1:
            weights = weights.view(-1, 1)
        else:
            phi = phi.unsqueeze(0) #add batch dim
            weights = weights.view(weights.size(0), weights.size(1), 1)

        score = weights*phi
        if self.up:
            score = score.transpose(-2,-1)
         
        y = torch.matmul(score,z) + x

        return y

#class to pool and unpool    
class Pool(nn.Module):  
    def __init__(
            self,
            hidden_dim,
            phi,
            prior_coef,
            dropout = 0,
            use_norm = False,
            reset = True,
            use_att = False,
            Skip = True,
            sqrt = True,
            up = False,
            ):
        super(Pool, self).__init__()
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        
        dim = in_channels if up else out_channels
            
        self.use_att = use_att
        self.phi = phi
        self.Skip = Skip
        self.sqrt = sqrt
        self.up = up
        
        
        self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
        self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
        self.norm = (lambda x: torch.nn.functional.normalize(x, p=1, dim=1)) if use_att else nn.Identity()
        self.norm1 = nn.LayerNorm(in_channels) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.weights = nn.Parameter(torch.Tensor(dim))
        
        if reset:
            self.reset_parameters()
        nn.init.constant_(self.weights, prior_coef)

        
    def forward(self, x):
        #sqrt normalization for cross attention like in "attention is all you need" but for crossattention
        n = torch.sqrt(torch.tensor(self.key.size(-1))) if self.sqrt else 1
        
        #layer normailzation
        if isinstance(self.norm1, nn.LayerNorm):
            x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        #actual pooling (crossattention if use_att = True)
        score = torch.matmul(self.key,self.query)/n
        score = self.dropout(self.norm(score)).T
        
        #Skip connection which can be disabled by Sk
        if self.Skip:
            if self.up:
                score = self.weights*self.phi + (1-self.weights)*score
            else:
                score = self.weights*self.phi.T + (1-self.weights)*score.T
                score = score.T

        y = torch.matmul(score,x)

        return y
    
    def reset_parameters(self):
        
        torch.nn.init.xavier_uniform_(self.key, gain=1.0)
        torch.nn.init.xavier_uniform_(self.query, gain=1.0)

#class for "conv" operator            
class Conv(nn.Module):      
    def __init__(
            self,
            eig_vals,
            use_eigenvals,
            conv_size,
            use_activation,
            prior_coef,
            activation='ReLU',
            dropout = 0,
            use_norm = False,
            bias = True,
            Skip = True,
            ):
        super(Conv, self).__init__()

        d = 4 if use_eigenvals else 3
        
        if activation == "SwiGLU":
           conv_size_in = int(conv_size/3)
           conv_size_out = conv_size_in*2
        else:
            conv_size_in = conv_size
            conv_size_out = conv_size
            
        activation = activation_func(activation)
        
        self.Skip = Skip
        self.eigv = eig_vals
        self.use_eigenvals = use_eigenvals
        self.l1 = nn.Linear(d, conv_size_out ,bias)
        self.l2 = nn.Linear(conv_size_in, 3 ,bias)
        self.dropout = nn.Dropout(p=dropout)                             
        self.activation = activation if use_activation == 1 else nn.Identity()
        self.norm = nn.LayerNorm(d) if use_norm else nn.Identity()
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        nn.init.constant_(self.weight, prior_coef)
        
    def forward(self, y):

        z = y
        
        if self.use_eigenvals:
            ev =torch.unsqueeze(self.eigv.repeat(y.size(0),1),2)
            y = torch.cat((y,ev),2)
            del ev
            
        y = self.norm(y)
        y = self.l1(y)
        y = self.activation(y)
        y = self.l2(y)
        y = self.dropout(y)            

        if self.Skip:
            z = (1-self.weight)*y + self.weight*z
        else:
            z = y

        return z
        

class SkipBlock_simple(nn.Module):
    def __init__(
            self,
            hidden_dim,
            conv_size,
            phi,
            eig_vals,
            prior_coef,
            activation='ReLU',
            dropout = 0,
            use_activation = 0,
            use_conv = True,
            use_eigenvals = False,
            use_norm = False,
            bias = True,
            reset = True,
            use_att = False,
            Skip = 2,
            sqrt = True, 
            up = False,
            use_weights = False,
            gate = False,
            ):
        super(SkipBlock_simple, self).__init__()
        
        ev_size = phi.size(1) if up else phi.size(0)
        
        self.pool = Pool(
                            hidden_dim,
                            phi,
                            prior_coef,
                            dropout,
                            use_norm,
                            reset,
                            use_att,
                            True if Skip >=2 else False,
                            sqrt,
                            up,
                        )
        
        self.conv = Conv(
                            eig_vals[:ev_size],
                            use_eigenvals,
                            conv_size,
                            use_activation,
                            prior_coef,
                            activation,
                            dropout,
                            use_norm,
                            bias,
                            True if 1<= Skip <= 2 else False,
                        )
        self.activation = activation_func(activation) if use_activation == -1 else nn.Identity()

        self.up = up
        self.use_weights  = use_weights 
        self.phi = phi
        self.Skip = Skip
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        nn.init.constant_(self.weight, prior_coef)  
        

    def forward(self, x, weights):
        
        
        if self.up:
            
            if self.use_weights: 
                x = (1-weights.view(-1, weights.size(-1), 1))*x
                
            z = self.conv(x)
            z = self.activation(z)
            z = self.pool(z)
        else:
            z = self.pool(x)
            z = self.activation(z)
            z = self.conv(z)
            if self.use_weights: 
                z = (1-weights.view(-1, weights.size(-1), 1))*z
        
        if self.Skip == -1:
            if self.phi.size(0) == self.phi.size(1):
                #spectral skip connection only if down or upsampling
                y = x
            else:
                y = torch.matmul(self.phi,x) 
            z = (1-self.weight)*z + self.weight*y  
        torch.cuda.empty_cache()
        return z
    


class SDL_skip_simple(nn.Module):
    def __init__(self, opt,eig_vecs,eig_vals):
        super().__init__()
        
        k = len(opt["nb_freq"])
        
        phi = [torch.tensor(eig_vecs[:,:opt["nb_freq"][i]],device = opt["device"],requires_grad=False) for i in range(k)]
                            
        Spectral_M_down = [phi[0].t()]
        for i in range(len(phi)-1):
            Spectral_M_down.append(torch.matmul(phi[i+1].t(),phi[i]))
        
        
        eig_vals = torch.tensor(eig_vals,device = opt["device"],requires_grad=False)      
        M = len(Spectral_M_down)
        
        self.SkipBlocksDown = nn.ModuleList(
            [
            SkipBlock_simple(
                hidden_dim = opt["hidden_dim"][i],
                conv_size = opt["conv_size"][i],
                phi = Spectral_M_down[i],
                eig_vals = eig_vals,
                prior_coef = opt["prior_coef"],
                activation = opt["activation"],
                dropout = opt["dropout"] if 'dropout' in opt else 0,
                use_activation = opt["use_activation"][i],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                up = False,
                use_weights = True if i == M-1 else False

                
            )
            for i in range(M)]
        )
        
        self.SkipBlocksUp = nn.ModuleList(
            [
            SkipBlock_simple(
                hidden_dim = opt["hidden_dim"][-i-1],
                conv_size = opt["conv_size"][-i-1],
                phi = Spectral_M_down[-i-1].T,
                eig_vals = eig_vals,
                prior_coef = opt["prior_coef"],
                activation = opt["activation"],
                dropout = opt["dropout"] if 'dropout' in opt else 0,
                use_activation = opt["use_activation"][i],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                up = True,
                use_weights = True if i == 0 else False

                
            )
            for i in range(M)]
        )
        
        self.gate = opt["gate"]
        if opt["gate"]:
            self.lD = nn.Linear(3*phi[-2].size(1), phi[-1].size(1) ,True)
            self.actD = nn.Sigmoid()
            self.lU = nn.Linear(3*phi[-1].size(1), phi[-1].size(1) ,True)
            self.actU = nn.Sigmoid()
        else:
           self.weightsD = nn.Parameter(torch.Tensor(phi[-1].size(1)))
           self.weightsU = nn.Parameter(torch.Tensor(phi[-1].size(1)))
           nn.init.constant_(self.weightsD, opt["prior_coef"])
           nn.init.constant_(self.weightsU, opt["prior_coef"])
        
        self.SkipDown = Skip(phi = phi[-1],
                             up = False,
                             )
        self.SkipUp = Skip(phi = phi[-1].T,
                           up = True,

                           )
        
        del phi
        del Spectral_M_down
        torch.cuda.empty_cache()


    def encoder(self, x):
        
        z = x
        i=1        
        for SB in  self.SkipBlocksDown:
            if i == len(self.SkipBlocksDown):
                
                if self.gate:
                    weights = self.actD(self.lD(torch.flatten(x,start_dim = 1)))
                else:
                    weights = self.weightsD
                    
                x = SB(x,weights)
                
            else:    
                x = SB(x,None)
            i+=1
            
        x = self.SkipDown(x,z,weights)
        
        return x
    
    def decoder(self, x):
        
        if self.gate:
            weights = self.actU(self.lU(torch.flatten(x,start_dim = 1)))
        else:
            weights = self.weightsU

        z = x        
        for SB in  self.SkipBlocksUp:
            x = SB(x,weights)
            
        x = self.SkipUp(x,z,weights)
        
        return x
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    