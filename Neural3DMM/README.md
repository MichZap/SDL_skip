# Neural3DMM/Deep3DMM

This is the implementation of Neural3DMM with attention pooling. 
All credit and copyright goes to [Neural3DMM](https://github.com/gbouritsas/Neural3DMM) for the Model and [Deep3DMM](https://github.com/zchen06/Deep3DMM) for the attention pooling approach and their licenses apply.



Tested with Python 3.8.1.6 and Anaconda on Windows 10
'''
Commands to install requirements in Anaconda:
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pandas
conda install -c anaconda scikit-learn
conda install -c conda-forge plyfile
'''


Due to incompabilities with the package versions required for the model on windows, we create the down and upsampling matrices externally in another enviroment (Mesh_downsampling.py).
In Anconda with python 3.10 on Windows. psbody.mesh can be installed:


'''
conda install git
pip install --no-cache-dir pyopengl==3.1.5
python -c "import urllib.request ; urllib.request.urlretrieve('https://github.com/johnbanq/psbody-mesh-build-script/releases/latest/download/install_psbody.pyz', 'install_psbody.pyz')" && python install_psbody.pyz
'''

Model can be trained by running main.py. Depending on the data we recommend running procrustes_ply(Data,folder) from SDL_skip/utilis beforehand to align the data.






