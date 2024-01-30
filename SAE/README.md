# SAE

This is the implementation of Spectral Autoencoder SAE.
Main changes:
- added hyperparameter to control down- and upsample-size
- change from pztorch lighting to pztorch implementation only
 
All credit and copyright goes to [SAE](https://github.com/MEPP-team/SAE) and their licenses apply.



Tested with Python 3.10.10 and Anaconda on Windows 10
Commands to install requirements in Anaconda:
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge polyscope
conda install pandas
conda install -c anaconda scikit-learn
conda install -c conda-forge plyfile
```


Model can be trained by running main.py. Depending on the data we recommend running procrustes_ply(Data,folder) from SDL_skip/utilis beforehand to align the data.






