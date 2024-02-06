import torch
import torch.nn as nn
import math
import copy

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
        s = x if  i==0 else xs[i-1]
        c = List[l+1](List[l](s))
        
        j=0
        while j < h[i]:
            l = int(math.floor(l/2))-min(j,1)
            s = x if  l in P else xs[i-j-1]
            b = List[l](s)
            c = c+b
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
    def __init__(self,hidden_dim,in_channels,out_channels,p,use_norm,value=True,sqrt=True):
        super(Att, self).__init__()
        
        
        self.key =  nn.Parameter(torch.Tensor(in_channels, hidden_dim))
        self.query = nn.Parameter(torch.Tensor(hidden_dim, out_channels))
        self.value = nn.Linear(3, 3,bias = False) if value else nn.Identity()
        self.norm = (lambda x: torch.nn.functional.normalize(x, p=1, dim=1)) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(p=p)
        self.sqrt = sqrt
        self.reset_parameters()
        
    def forward(self,x):
        
        n = torch.sqrt(torch.tensor(self.key.size(-1))) if self.sqrt else 1
        score = torch.matmul(self.key,self.query)/n
        score = self.norm(score)
        v = self.value(x)
        
        return torch.matmul(self.dropout(score).T,v)
        
    def reset_parameters(self):
        
        torch.nn.init.xavier_uniform_(self.key)
        torch.nn.init.xavier_uniform_(self.query)
        if isinstance(self.value, nn.Linear):
            torch.nn.init.xavier_uniform_(self.value.weight)

        
class MultiHeadAtt(nn.Module):
    def __init__(self,hidden_dim,n_heads,p,use_norm,phi,value=True,sqrt=True):
        super(MultiHeadAtt, self).__init__()
        
        in_channels = phi.size(1)
        out_channels = phi.size(0)
        hd = int(hidden_dim/n_heads)
        
        self.phi = phi 
        self.heads = nn.ModuleList([Att(hd,in_channels,out_channels,p,use_norm,value,sqrt) for _ in range(n_heads)])
        self.value = nn.Linear(3, 3,bias = False) if value else nn.Identity()
        self.proj = nn.Linear(3 * (n_heads+1), 3)
        self.dropout = nn.Dropout(p=p)
        if value:
            self.reset_parameters()
    
    def forward(self,x):
    
        heads = [h(x) for h in self.heads]   
    
        v = self.value(x) 
        phi_v = torch.matmul(self.phi,v)
        heads.append(phi_v)
        
        mheads = torch.cat(heads, dim=-1)
        proj = self.dropout(self.proj(mheads))
    
        return proj
			

    def reset_parameters(self):
        
        torch.nn.init.xavier_uniform_(self.value.weight)
