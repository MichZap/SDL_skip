# SDL_skip

Official repository for the  paper Deep learnable spectral decomposition of baby faces.
We also include slightly modified versions of SAE and Deep3DMM/Neural3DMM, both used in the paper.

## Abstract

*In this paper, we introduce a novel, deep 3D morphable model for meshes with common
triangulation. Specifically, we apply it to reconstruct baby faces. The proposed
algorithm is simple, adaptable, and specifically targeted to perform well on small
datasets. We combine Graph-Laplacian based spectral decomposition with a learnable,
transformer-like component. The decomposition matrices are applied as skipconnections,
providing our architecture with a prior that encodes both local and global
information of the underlying mesh structure. The learnable component does not make
any domain-specific assumptions and can override the prior, if necessary. This flexibility
also allows our model to perform well on larger datasets. We further modify the
decomposition matrices to create deeper versions of this architecture and introduce a
data augmentation strategy: flipping and rotations are applied to the deviations from
the mean, rather than directly to the samples. In our experiments, we compare the reconstruction
error of the proposed architecture against the state of the art, examine the
effect of data augmentation across a small baby face dataset and a larger adult dataset
and inspect our model’s capabilities to generate new samples from the encoded distribution.
We show that our method outperforms current baby face models, as well
as state of the art 3D morphable models, especially on the raw data. Additionally,
we demonstrate that the proposed data augmentation substantially improves existing
models.*


## Requirements
Requirements depend on the model and are specified in the respective folders.
Deep3DMM/Neural3DMM require to precalculate the pooling matrices based on minimizing squared
error distance. On Windows it is difficult if not impossible to have both the code for calculating these matrices as well Deep3DMM/Neural3DMM running under the same python version.
We different enviroments for that.


## Acknowledgement
The implementation is based on the code of SAE and Deep3DMM/Neural3DMM are based on their originals:

Many thanks to the authors for releasing the source code.
