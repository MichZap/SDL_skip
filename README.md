# SDL_skip
Deep spectral decomposition

Official repository for the FG2024 paper Deep learnable spectral decomposition on baby faces
We also include slightly modified versions of SAE and Deep3DMM/Neural3DMM, both used in the paper.

## Requirements
Requirements depend on the model and are specified in the respective folders.
Deep3DMM/Neural3DMM require to precalculate the pooling matrices based on minimizing squared
error distance. On Windows it is difficult if not impossible to have both the code for calculating these matrices as well Deep3DMM/Neural3DMM running under the same pyzthon version.
We different enviroments for that.


## Acknowledgement
The implementation is based on the code of SAE and Deep3DMM/Neural3DMM are based on their originals:

Many thanks to the authors for releasing the source code.
