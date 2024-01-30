# SDL_skip

Official repository for the FG2024 paper Deep learnable spectral decomposition on baby faces.
We also include slightly modified versions of SAE and Deep3DMM/Neural3DMM, both used in the paper.

## Abstract

*Autoencoders on meshes mostly use precal-
culated pooling and upsampling in combination with spatial
or spectral convolutions. In this paper, we propose a simple,
flexible yet effective algorithm for meshes with common topol-
ogy, combining Graph-Laplacian based spectral decomposition
with a transformer-like learnable component. Applying the
decomposition matrices as skip-connection, provides our archi-
tecture with a prior, that contains local and global information
with regards to the underlying mesh structure. The learnable
component does not make any domain specific assumptions and
is able to overwrite the prior, if needed. This allows our model
to perform well on both small and large datasets. We further
manipulate the decomposition matrices to generate deeper
versions of this architecture.*


## Requirements
Requirements depend on the model and are specified in the respective folders.
Deep3DMM/Neural3DMM require to precalculate the pooling matrices based on minimizing squared
error distance. On Windows it is difficult if not impossible to have both the code for calculating these matrices as well Deep3DMM/Neural3DMM running under the same python version.
We different enviroments for that.


## Acknowledgement
The implementation is based on the code of SAE and Deep3DMM/Neural3DMM are based on their originals:

Many thanks to the authors for releasing the source code.
