import torch
import torch.nn as nn

import pdb
from attpool import AttPool

class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size,out_c,activation='elu',bias=True,device=None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x,spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
  
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device=self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat

class SpiralAutoencoder(nn.Module):
    def __init__(self, sizes, spiral_sizes, spirals, D, U, V_ref=None,config=None):
        super(SpiralAutoencoder,self).__init__()
        self.latent_size = config["size_latent"]
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = config["filters_enc"]
        self.filters_dec = config["filters_dec"]
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = config["device"]
        self.activation = config["activation"]
        self.upsample_att = config["upsample_att"]
        self.downsample_att = config["downsample_att"]
        
        self.conv = []
        input_size = self.filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if self.filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], self.filters_enc[1][i],
                                            activation=self.activation, device=self.device).to(self.device))
                input_size = self.filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], self.filters_enc[0][i+1],
                                        activation=self.activation, device=self.device).to(self.device))
            input_size = self.filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, self.latent_size)
        self.fc_latent_dec = nn.Linear(self.latent_size, (sizes[-1]+1)*self.filters_dec[0][0])
        
        self.dconv = []
        input_size = self.filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], self.filters_dec[0][i+1],
                                             activation=self.activation, device=self.device).to(self.device))
                input_size = self.filters_dec[0][i+1]  
                
                if self.filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], self.filters_dec[1][i+1],
                                                 activation=self.activation, device=self.device).to(self.device))
                    input_size = self.filters_dec[1][i+1]
            else:
                if self.filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], self.filters_dec[0][i+1],
                                                 activation=self.activation, device=self.device).to(self.device))
                    input_size = self.filters_dec[0][i+1]                      
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], self.filters_dec[1][i+1],
                                                 activation='identity', device=self.device).to(self.device)) 
                    input_size = self.filters_dec[1][i+1] 
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], self.filters_dec[0][i+1],
                                                 activation='identity', device=self.device).to(self.device))
                    input_size = self.filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)
        
        if max(self.upsample_att) is True or max(self.downsample_att) is True:
            self.attpoolenc, self.attpooldec = self.init_attpool(config, V_ref)
            self.attpoolenc = torch.nn.ModuleList(self.attpoolenc)
            self.attpooldec = torch.nn.ModuleList(self.attpooldec)
    
    def init_attpool(self, config, V_ref=None):
        attpooldec = []
        attpoolenc = []
        if V_ref is not None:
            for i in range(len(V_ref)-1):
                idx_k, idx_q = i, i+1
                k_init, q_init = V_ref[idx_k].t(), V_ref[idx_q].t()
                config_enc = config
                config_enc['num_k'], config_enc['num_q'] = k_init.size(1), q_init.size(1)
                config_enc['top_k'] = config['top_k_enc'][min(i,len(config['top_k_enc'])-1)] \
                    if isinstance(config['top_k_enc'],list) else config['top_k_enc']
                attpoolenc.append(AttPool(config=config_enc, const_k=k_init, const_q=q_init,
                                          prior_map=self.D[i]))
                config_dec = config
                config_dec['num_k'], config_dec['num_q'] = q_init.size(1), k_init.size(1)
                config_dec['top_k'] = config['top_k_dec'][min(i,len(config['top_k_dec'])-1)] \
                    if isinstance(config['top_k_dec'],list) else config['top_k_dec']
                attpooldec.append(AttPool(config=config_dec, const_k=q_init, const_q=k_init,
                                          prior_map=self.U[i]))
        return attpoolenc, attpooldec

    def encode(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            if self.downsample_att[i] is True:
                x = self.attpoolenc[i](x)
            else:
                x = torch.matmul(D[i],x)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            if self.upsample_att[i] is True:
                x = self.attpooldec[-i-1](x)
            else:
                x = torch.matmul(U[-1-i],x)
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
                j+=1
        return x

    
    def forward(self,x):
        z = self.encode(x)
        x = self.decode(z)
        return x   

