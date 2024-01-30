#Use the conda enviroment with python 3.10 and and psbody.mesh installed.
#Due to incompabilities with the package versions required for the coma model, we create the down and upsampling matrices externally, save them and then load them into another enviornment.

#To install in conda run:
#
#    conda install git
#    pip install --no-cache-dir pyopengl==3.1.5
#    python -c "import urllib.request ; urllib.request.urlretrieve('https://github.com/johnbanq/psbody-mesh-build-script/releases/latest/download/install_psbody.pyz', 'install_psbody.pyz')" && python install_psbody.pyz

from psbody.mesh import Mesh
import mesh_sampling_mod as mo
import os
import pickle  

#input link
template_file_path=r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\FaceDataWarehouse\Mean_mesh_FW2.ply"#ply reformated with PlyData text = False
folder=r"C:\Users\Michael\PhD_MZ\Autoencoder Babyface\Data\FaceDataWarehouse"

template_mesh = Mesh(filename=template_file_path)
M, A, D, U, F = mo.generate_transform_matrices(template_mesh, [4,4,4,4])

with open(os.path.join(folder,'downsampling_matrices.pkl'), 'wb') as fp:
        M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
        pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)