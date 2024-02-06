import torch
import torch.nn as nn
from model_utilis import  activation_func

class AttBlock(nn.Module):
    def __init__(
            self,
            hidden_dim,
            phi,
            prior_coef,
            dropout = 0,
            use_norm = False,
            use_att = False,
            Skip = 2,
            sqrt = True, 
            up = False
            ):
        super(AttBlock, self).__init__()    
    
class Skip(nn.Module):
    def __init__(
            self,
            phi,
            prior_coef,
            up = False,
            ):
        super(Skip, self).__init__()
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        
        dim = in_channels if up else out_channels
        
        self.phi = phi
        self.up = up
        self.weights = nn.Parameter(torch.Tensor(dim))
        nn.init.constant_(self.weights, prior_coef)
        
    def forward(self, x, z):
        
        if self.up:
            score = self.weights*self.phi
            score = score.T
        else:
            score = self.weights*self.phi.T

        y = torch.matmul(score,z) + torch.mean(self.weights)*x
        
        return y    
        

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
            up = False
            ):
        super(SkipBlock_simple, self).__init__()
        
        
        d = 4 if use_eigenvals else 3
        
        if activation == "SwiGLU" and use_activation == -1:
           conv_size_in = int(conv_size/3)
           conv_size_out = conv_size_in*2
        else:
            conv_size_in = conv_size
            conv_size_out = conv_size
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        
        dim = in_channels if up else out_channels
        
        activation = activation_func(activation)
            
        self.use_att = use_att
        self.phi = phi
        self.eigv = eig_vals[:out_channels]
        self.use_eigenvals = use_eigenvals
        self.Skip = Skip
        self.sqrt = sqrt
        self.up = up
        

        self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
        self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
        self.norm = (lambda x: torch.nn.functional.normalize(x, p=1, dim=1)) if use_att else nn.Identity()        
        self.l3 = nn.Linear(d, conv_size_out ,bias) if use_conv else nn.Identity()
        self.l4 = nn.Linear(conv_size_in, 3 ,bias) if use_conv else nn.Identity()                               
        self.activation = activation if use_activation >= 1 else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(in_channels) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d) if use_norm else nn.Identity()
        self.weights = nn.Parameter(torch.Tensor(dim))
        self.weight = nn.Parameter(torch.Tensor(1))

        
        if reset:
            self.reset_parameters()
        nn.init.constant_(self.weights, prior_coef)
        nn.init.constant_(self.weight, prior_coef)
        
            

    def forward(self, x):
        

        #IW = self.I-self.W

        n = torch.sqrt(torch.tensor(self.key.size(-1))) if self.sqrt else 1
        
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        score = torch.matmul(self.key,self.query)/n
        score = self.dropout(self.norm(score)).T
        
        if self.Skip >= 2:
            if self.up:
                score = self.weights*self.phi + (1-self.weights)*score
            else:
                score = self.weights*self.phi.T + (1-self.weights)*score.T
                score = score.T


        y = torch.matmul(score,x)
        
        if 1<= self.Skip <= 2:
            z = y

            
        if self.use_eigenvals:
            ev =torch.unsqueeze(self.eigv.repeat(x.size(0),1),2)
            y = torch.cat((y,ev),2)
            del ev
        y = self.norm2(y)
        y = self.l3(y)
        y = self.activation(y)
        y = self.l4(y)
        y = self.dropout(y)            

        if 1<= self.Skip <= 2:
            z = (1-self.weight)*y + self.weight*z
        else:
            z = y               
        
        del y
        torch.cuda.empty_cache()
        
        return z
    
    def reset_parameters(self):
        
        torch.nn.init.xavier_uniform_(self.key, gain=1.0)
        torch.nn.init.xavier_uniform_(self.query, gain=1.0)

        #if isinstance(self.l3, nn.Linear):
        #    torch.nn.init.xavier_uniform_(self.l3.weight, gain=1.0)
        #    torch.nn.init.xavier_uniform_(self.l4.weight, gain=1.0)


class SDL_skip_simple(nn.Module):
    def __init__(self, opt,eig_vecs,eig_vals):
        super().__init__()
        
        k = len(opt["nb_freq"])
        
        phi = [torch.tensor(eig_vecs[:,:opt["nb_freq"][i]],device = opt["device"],requires_grad=False) for i in range(k)]
                            
        Spectral_M_down = [phi[0].t()]
        for i in range(len(phi)-1):
            Spectral_M_down.append(torch.matmul(phi[i+1].t(),phi[i]))
        
        
        eig_vals = torch.tensor(eig_vals,device = opt["device"],requires_grad=False)

        self.SkipDown = Skip(phi = phi[-1],prior_coef = opt["prior_coef"],up = False)
        self.SkipUp = Skip(phi = phi[-1].T,prior_coef = opt["prior_coef"],up = True)
        
        del phi
        torch.cuda.empty_cache()
        
        
        
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
                use_activation = opt["use_activation"],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                up = False,
                
            )
            for i in range(len(Spectral_M_down))]
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
                use_activation = opt["use_activation"],
                use_conv = opt["use_conv"],
                use_eigenvals = opt["use_eigenvals"],
                use_norm = opt["use_norm"] if 'use_norm' in opt else False,
                bias = opt["bias"],
                reset = opt["reset"] if "reset" in opt else False,
                use_att = opt["use_att"] if "use_att" in opt else False,
                Skip = opt["Skip"] if "Skip" in opt else 2,
                sqrt = opt["sqrt"] if "sqrt" in opt else True,
                up = True,
                
            )
            for i in range(len(Spectral_M_down))]
        )
        del Spectral_M_down
        torch.cuda.empty_cache()


    def encoder(self, x):
        
        z = x        
        for SB in  self.SkipBlocksDown:
            x =SB(x)
            
        x = self.SkipDown(x,z)
        
        return x
    
    def decoder(self, x):
        
        z = x        
        for SB in  self.SkipBlocksUp:
            x =SB(x)
            
        x = self.SkipUp(x,z)
        
        return x
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    