import random
import torch
import numbers
import math

#transforms based on pytrochgeometric fucntions
class RandomFlip(object):
    """Flips node positions along a given axis randomly with a given
    probability (functional name: :obj:`random_flip`).

    Args:
        axis (int): The axis along the position of nodes being flipped.
        p (float, optional): Probability that node positions will be flipped.
            (default: :obj:`0.5`)
    """
    def __init__(self, axis: int, p: float = 0.5):
        self.axis = axis
        self.p = p
    
    def __call__(self, verts):
        #print("flip: ",self.axis)
        if random.random() < self.p:
            pos = verts.clone()
            pos[..., self.axis] = -pos[..., self.axis]
            verts = pos
        return verts
    
class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix

    def __call__(self, verts):
        pos = verts.clone()
        pos = pos.view(-1, 1) if pos.dim() == 1 else pos

        assert pos.size(-1) == self.matrix.size(-2), (
            'Node position matrix and transformation matrix have incompatible '
            'shape.')

        pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))
        verts = pos

        return verts

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())

class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees=180, axis=0, p=0.5):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.p = p

    def __call__(self, verts):
        #print("rot: ",self.axis)
        if random.random() < self.p:
            pos = verts.clone()
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)

            if pos.size(1) == 2:
                matrix = [[cos, sin], [-sin, cos]]
            else:
                if self.axis == 0:
                    matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
                elif self.axis == 1:
                    matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
                else:
                    matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
            return LinearTransformation(torch.tensor(matrix))(pos)
        else:
            return verts

    def __repr__(self):
        return '{}({}, axis={})'.format(self.__class__.__name__, self.degrees,
                                        self.axis)
    
class AddGaussianNoise(object):
    """Adds Gaussian noise to vertex positions.

    Args:
        sigma (float): Standard deviation of the Gaussian noise.
    """
    def __init__(self, sigma: float, p= 0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, verts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            verts (torch.Tensor): Tensor of shape (..., D) representing vertex coordinates.

        Returns:
            torch.Tensor: Noisy vertices of the same shape.
        """
        if random.random() < self.p:
            
            # Draw noise from N(0, sigma^2) for each coordinate
            noise = torch.randn_like(verts) * self.sigma
            return verts + noise
        else:
            return verts
 
def rename(path,mean,sd,norm,sym):
    if mean is not None:
        path = path[:-3] + "_mean.pt"
    
    if sd is not None:
        path = path[:-3] + "_sd.pt"
        
    if norm:
        path = path[:-3] + "_norm.pt"
        
    if sym == "sym" or sym == "asym":
        path = path[:-3] + "_sym.pt"

    return path        

class SpectralInterpolation(object):
    def __init__(self, dataset, phi, k, root_dir, mu=0.5, sigma=0.5, p=0.5, mean = None, sd = None, norm = None, sym =None):
        self.dataset = dataset
        self.phi = torch.tensor(phi.T).float()
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.root_dir = root_dir
        
        self.mean = mean
        self.sd = sd
        self.norm = norm
        self.sym = sym

    def __call__(self, verts):
        #verts not needed take two random meshes and spectracl interpolate
        if random.random() < self.p:
            
            phi_k = self.phi[:self.k,:]

            
            idx = random.randrange(len(self.dataset))
            idx2 = random.randrange(len(self.dataset))
            
            path = self.root_dir+"\\processed\\"+self.dataset.Name.iloc[idx]+".pt"
            path2 = self.root_dir+"\\processed\\"+self.dataset.Name.iloc[idx2]+".pt"
            
            path = rename(path,self.mean,self.sd,self.norm,self.sym)
            path2 = rename(path2,self.mean,self.sd,self.norm,self.sym)
            
            X1 = torch.load(path)['points']
            X2 = torch.load(path2)['points']

            
            delta = X2 - X1
            rho = torch.normal(mean=self.mu, std=self.sigma, size=(self.k,))

            # Create a diagonal matrix from the vector
            diag = torch.diag(rho)# k×k

            offset = phi_k.T @ (diag @ (phi_k @ delta))
            verts = X1 + offset

        return verts