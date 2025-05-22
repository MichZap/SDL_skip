import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F

def activation_func(activation):
    
        if activation == "ReLU":
            activation = nn.ReLU()#causes meomory leak?
        elif activation == "Tanh":
            activation = nn.Tanh()
        elif activation == "Sigmoid":
            activation = nn.Sigmoid()
        elif activation == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif activation == "ELU":
            activation = nn.ELU()
        elif activation == "SiLU":
            activation = nn.SiLU()
        elif activation == "SwiGLU":
            activation = SwiGLU()#only implemented for the second layer
        elif activation == "identity":
            activation = nn.Identity()
        else:
            raise ValueError(f"Invalid activation function: {activation}")
            
        return activation

def generate_series(n):
    #generates a series which repeats it self but for the last entry always 1 is added:
    #1
    #1 2
    #1 2 1 3
    #1 2 1 3 1 2 1 4
    #this gives information on how many skip connections need to be used in the skipping pyramid
    
    s = [1]

    for i in range(n):
        s2 = s.copy()
        k = len(s)-1
        s2[k] = s2[k] +1
        
        s.extend(s2)

    return s


def generate_inverse_series(n):
    #idea is to create index list in order to get the spectral upsampling matrices without explcitly needing to calculate them and save cuda memory
    series = list(range(n))  # Create the range(n) list

    i = 0
    s = 0
    while s < n:
        j = min(s + 2 ** i, n)  # Calculate the end index for inversion
        series[s:j] = reversed(series[s:j])  # Inverse the elements in the range
        s = j
        i = i+1

    return series

def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return PE



def IterativeSkip(x,List):
    
    n = len(List)
    ll= int(math.floor(n/2))
    
    if n ==1:
        return List[0](x)
    
    #calculate deepth of the skip pyramid
    m = copy.copy(ll)
    v = 1
    while m >= 1:
        v = v +1
        m = int(math.floor(m/2))

    h = generate_series(v-1)
    P = [int(math.floor(n/2**i)) for i in range(max(h)+1)]
    
    xs = []
    
    for i in range(int(math.ceil(ll/2))):
        l = ll+2*i
        s = x if  i==0 else xs[i-1][0]
        c = List[l+1](List[l](s)[0])
        
        j=0
        while j < h[i]:
            l = int(math.floor(l/2))-min(j,1)
            s = x if  l in P else xs[i-j-1][0]
            b = List[l](s)

                
            c = (c[0]+b[0],b[1])
            j = j+1
        xs.append(c)
        
    del xs
    torch.cuda.empty_cache()
    
    return c


#implementation from PaLM
class SwiGLU(nn.Module):
    def forward(self,x):
        x, gate = x.chunk(2,dim = -1)
        return nn.functional.silu(gate)*x


class Att(nn.Module):
    def __init__(self,hidden_dim,in_channels,out_channels,p,use_norm_att,value=True,sqrt=True,use_att=True,flatten=False,bias=False,reset=False,vinit=False,eb_dim=None):
        super(Att, self).__init__()
        
        if use_att:
            if vinit == False:
                self.key = nn.Parameter(torch.empty(in_channels, hidden_dim))
                self.query = nn.Parameter(torch.empty(hidden_dim, out_channels))
            else:
                if eb_dim:
                    self.eb1 = nn.Linear(6,eb_dim,bias)
                    self.eb2 = nn.Linear(6,eb_dim,bias)
                else:
                    self.eb1 = nn.Identity()
                    self.eb2 = nn.Identity()
                    eb_dim = 6
                self.l1 = nn.Linear(eb_dim, hidden_dim,bias)
                self.l2 = nn.Linear(eb_dim, hidden_dim,bias)
                self.lv = nn.Linear(eb_dim, 3,bias) if value == True else nn.Identity()
                
            if use_norm_att == True:
                self.norm = (lambda x: torch.nn.functional.normalize(x, p=1, dim=1))
            elif use_norm_att == "Soft": 
                self.norm = nn.Softmax(dim=1)
            else:
                self.norm = nn.Identity()
            self.sqrt = sqrt
            
        else:
             self.l1 = nn.Linear(in_channels, hidden_dim,bias)
             self.l2 = nn.Linear(hidden_dim, out_channels,bias)
             if flatten:
                 self.l3 = nn.Linear(3*hidden_dim, 3*hidden_dim,bias)
                 
        self.value = nn.Linear(3, 3,bias) if value and eb_dim == None else nn.Identity()
        self.dropout = nn.Dropout(p=p)
        self.use_att = use_att
        self.flatten = flatten
        if reset:
            self.reset_parameters()
        
    def forward(self,x,y=None):
        
        v = self.value(x)
        
        if self.use_att:
            if y is None:
                n = self.key.size(-1) ** 0.5 if self.sqrt else 1
                score = torch.matmul(self.key,self.query)/n
                score = self.norm(score)
                y = torch.matmul(self.dropout(score).T,v)
            else:
                x_emb = self.eb1(x)
                y_emb = self.eb2(y)
                query = self.l1(x_emb).transpose(-2, -1)
                key =   self.l2(y_emb)
                value = self.lv(x_emb) if isinstance(self.lv,nn.Linear) else x[:,:,:3]
                n = key.size(-1) ** 0.5 if self.sqrt else 1
                #with torch.cuda.amp.autocast(enabled=True):
                score = torch.matmul(key,query)/n
                score = self.norm(score)
                y = torch.matmul(self.dropout(score),value)
                
                #y = F.scaled_dot_product_attention(
                #       key,
                #       query.transpose(-2, -1),
                #       value,
                #       dropout_p=self.dropout.p
                #   )
            

            
        else:
            y = self.l1(v.permute(0, 2, 1))
            if self.flatten:
                shape = y.shape
                y = torch.flatten(y, start_dim=1)
                y = self.l3(y)
                y = y.view(shape)
                
            y = self.l2(y)
            y = self.dropout(y).permute(0, 2, 1)    

        
        return y
        
    def reset_parameters(self):
        
        if self.use_att and not hasattr(self, 'l1'):
            torch.nn.init.xavier_uniform_(self.key)
            torch.nn.init.xavier_uniform_(self.query)
            if isinstance(self.value, nn.Linear):
                torch.nn.init.xavier_uniform_(self.value.weight)
        else:     
            torch.nn.init.xavier_uniform_(self.l1.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.l2.weight, gain=1.0)
            if self.flatten:
                torch.nn.init.xavier_uniform_(self.l3.weight, gain=1.0)    

       

        
class MultiHeadAtt(nn.Module):
    def __init__(self,hidden_dim,n_heads,p,use_norm_att,phi,value=True,sqrt=True,use_att=True,flatten=False,Skip=False,bias=False,use_weights=False,up=False,reset=False):
        super(MultiHeadAtt, self).__init__()
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        hd = int(hidden_dim/n_heads)
        
        dim = in_channels if up else out_channels
        
        
        self.phi = phi 
        self.Skip = Skip
        self.heads = nn.ModuleList([Att(hd,in_channels,out_channels,p,use_norm_att,value,sqrt,use_att,flatten,bias,reset) for _ in range(n_heads)])
        self.value = nn.Linear(3, 3,bias) if value else nn.Identity()
        
        if Skip:
            n_heads = n_heads+1
        
        
        if use_weights:
            self.weights = nn.Parameter(torch.Tensor(n_heads,dim))
            nn.init.constant_(self.weights, 1/n_heads)
        else:
            self.proj = nn.Linear(3 * n_heads, 3) if Skip or n_heads >1 else nn.Identity()
        
        
        self.use_weights = use_weights
        self.up = up
        
        self.dropout = nn.Dropout(p=p)
        
        if reset:
            self.reset_parameters()
    
    def forward(self,x):
    

        if self.use_weights and self.up:

            weights = self.weights.unsqueeze(1).unsqueeze(3)
            heads = [h(weights[i]*x) for i,h in enumerate(self.heads)]

        else:
            heads = [h(x) for h in self.heads]
    
        v = self.value(x)
        
        if self.Skip:
            phi_v = torch.matmul(self.phi,v)
            heads.append(phi_v)
        
        if self.use_weights:
            if not self.up:
                weights = self.weights.unsqueeze(1).unsqueeze(3)
                weighted_heads = [heads[i] * weights[i] for i in range(len(heads))]
                proj = torch.sum(torch.stack(weighted_heads),dim = 0)
            else:
                proj = torch.sum(torch.stack(heads),dim = 0)
            proj = self.dropout(proj)
        else:
            mheads = torch.cat(heads, dim=-1)
            proj = self.dropout(self.proj(mheads))
    
        return proj
			

    def reset_parameters(self):
        
        if isinstance(self.value, nn.Linear):
            torch.nn.init.xavier_uniform_(self.value.weight)
        if isinstance(self.proj, nn.Linear):
            torch.nn.init.xavier_uniform_(self.proj.weight)

#simple implementation from Learning Local Neighboring Structure for Robust 3D Shape Representation
#their github seems too complicated 
class PaiConv(nn.Module):
    def __init__(self, Adj_mat, in_ch, out_ch, activation = "Elu", skip = False, small = False,k = 9 ):
        super(PaiConv, self).__init__()
        
        self.activation = activation_func(activation)
        
        
        self.small = small                          #indicates if we reduce the dimension of P, either False or reduced dim
        self.skip = skip                            #enables skip connection
        self.k = k                                  #length of the neighbourhood
        self.n = Adj_mat.size(0)                    #number of vertices
        self.in_ch = in_ch                          #in channels
        self.out_ch = out_ch                        #out channels
        self.indices = self.pad_trunc(Adj_mat)      #list of indices, including the neighbouring vertices for each vertex adjacency matrix truncated and padded
        
        if small == False:
            self.P = nn.Parameter(torch.Tensor(self.n,k,k))             #soft permutation matrix per vertex 
            self.P.data = torch.eye(self.k, device=self.P.device).unsqueeze(0).repeat(self.n, 1, 1)     #initialize P
        else:
            self.P = nn.Parameter(torch.Tensor(small,k,k))              #soft permutation matrix per vertex reduced dimension
            self.v = nn.Parameter(torch.Tensor(self.n,small))           #v*P has original dimension
            self.P.data = torch.eye(self.k, device=self.P.device).unsqueeze(0).repeat(small, 1, 1)      #initialize P
            self.v.data = torch.ones(self.n,small)/small

        self.conv = nn.Linear(k*in_ch, out_ch, bias = True)
        
        if skip:
            self.mlp_out = nn.Linear(in_ch, out_ch, bias = True)
    
    
    def pad_trunc(self,M):
        

        List = []

        for i in range(self.n):
            indices = M[i]
            indices = indices[indices>0]
            l = indices.sum()
            if l >= self.k:
                indices = indices[:self.k]
            else:
                pad_indices = torch.arange(self.n + 1, self.n + 1 + self.k - l, dtype=indices.dtype, device=indices.device)
                indices = torch.cat((indices, pad_indices))
            List.append(indices)
                
        return torch.stack(List)


    
    def forward(self,x):
        """
        Args:
            x: Input tensor of shape (batch_size, n, in_ch)
        Returns:
            z: Output tensor of shape (batch_size, n, out_ch)
        """
        #start_time = time.time()
        batch_size = x.size(0)  # Get batch size
        x_padded = torch.cat([x, torch.zeros(batch_size, self.k, self.in_ch, device=x.device)], dim=1)  # Shape: (batch_size, n + k, in_ch)
        indices = self.indices.to(torch.int64).to(x.device)
        
        
        batch_idx = torch.arange(batch_size, device=x.device)[:, None, None, None]                      # Creates tensor of shape (batch_size, 1, 1, 1) containing [0,1,2,...,batch_size-1]
        
        feat_idx = torch.arange(self.in_ch, device=x.device)[None, None, None, :]                       # Creates tensor of shape (1, 1, 1, in_ch) containing [0,1,2,...,in_ch-1]
        
        neighbours = x_padded[
        batch_idx,                                                                                      # Select which batch
        indices[None, :, :, None].expand(batch_size, self.n, self.k, self.in_ch),                       # Select which vertices
        feat_idx.expand(batch_size, self.n, self.k, self.in_ch)                                         # Select which features
        ]
        
        if self.small == False:
            permuted_neighbours = torch.matmul(self.P, neighbours)                                      # Shape: (batch_size, n, k, in_ch)
        else:
            P = torch.einsum('ns, skt->nkt', self.v, self.P)[None]                                                             #project reduced dim back to org dim
            permuted_neighbours = torch.matmul(P, neighbours)
            
        # Flatten and apply linear layer
        permuted_neighbours_flat = self.activation(permuted_neighbours.view(batch_size, self.n, -1))                     # Shape: (batch_size, n, k * in_ch)
        z = self.conv(permuted_neighbours_flat)
        z = self.activation(z)
        
        if self.skip:
            z = self.mlp_out(x) + z

        return z


class DyT(nn.Module):
    def __init__(self,C,init_a):
        super(DyT, self).__init__()

        self.y = nn.Parameter(torch.ones(C))  
        self.C = nn.Parameter(torch.zeros(C))
        self.a = init_a
    
    def forward(self,x):
        
        x = torch.tanh(self.a*x)
        x = self.y * x + self.C
        
        return x
            
        
#Architecture to learn a layers sensitivity regarding its input
#Model can learn which dimension of the input it derives
#general idea (layer(x+h)-layer(x))/h            
class Partial_Derivative(nn.Module):
    def __init__(self,layer,h,threshold,dimx,eps = 1e-8):
        super(Partial_Derivative, self).__init__()
        
        self.layer = layer #Input Layer which should be derived, can also be a model
        self.h = h #hyperparameter should be a small scalar
        self.threshold = threshold #threshold for weight masking 
        self.eps = eps #numerical paramater to prevent devivision by 0
        
        self.weights = nn.Parameter(torch.randn(dimx))
        self.dimx = torch.Tensor(dimx)

        
    def forward(self,x):

        batch_size = x.size(0)  # Get batch size
        # Compute a binary mask based on the threshold
        mask = (self.weights > self.threshold).float()  # 1 if active, 0 otherwise
        # Scale the input components with the masked weights
        h_scaled = self.h * (self.weights * mask)
        h_scaled_exp = h_scaled.expand(batch_size,h_scaled.size(0),h_scaled.size(1))
        

        y = (self.layer(x + h_scaled_exp) - self.layer(x))/(torch.linalg.norm(h_scaled ,ord=1) + self.eps)
         
        return y
