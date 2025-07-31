# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 20:20:31 2025

@author: Michael
"""

import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import os

from matplotlib import cm
from plyfile import PlyData, PlyElement
from scipy.linalg import orthogonal_procrustes


def load_meshlab_pp_xml(filepath):
    
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pts":
        with open(filepath, 'r') as f:
            lines = f.readlines()
    
        # Remove comments and empty lines
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('#') and not line.startswith('Version')]
    
        try:
            num_points = int(lines[0])
        except ValueError:
            raise ValueError(f"Invalid number of landmarks in {filepath}")
    
        if len(lines[1:]) < num_points:
            raise ValueError(f"Expected {num_points} points, but found {len(lines[1:])} in {filepath}")
    
        coords = [list(map(float, line.split()[1:])) for line in lines[1:num_points+1]]
        #names = [line.split()[0:1] for line in lines[1:num_points+1]]
        names = ["exR","enR","n","enL","exL","acR","aR","prn","aL","acL","sn","chR","cphR","ls","cphL","chL","li","sl","pg","tR","oiR","tL","oiL"]
    
        return np.array(coords), names

    elif ext == ".pp":
    
    
        tree = ET.parse(filepath)
        root = tree.getroot()
    
        landmarks = []
        names = []
    
        for point in root.findall('point'):
            x = float(point.attrib['x'])
            y = float(point.attrib['y'])
            z = float(point.attrib['z'])
            name = point.attrib.get('name', '')  # Some might not have a name
            landmarks.append([x, y, z])
            names.append(name)
    
        return np.array(landmarks), names



def align_meshes_by_landmarks(source_vertices, df):
    
    source_landmarks = df.iloc[:,:3]
    target_landmarks = df.iloc[:,4:]

    # Step 1: Center the landmarks
    src_mean = source_landmarks.mean(axis=0).values
    tgt_mean = target_landmarks.mean(axis=0).values

    src_centered = source_landmarks.values - src_mean
    tgt_centered = target_landmarks.values - tgt_mean

    # Step 2: Compute Procrustes rotation and scale
    R, _ = orthogonal_procrustes(src_centered, tgt_centered)
    scale = np.linalg.norm(tgt_centered, 'fro') / np.linalg.norm(src_centered, 'fro')

    # Step 3: Transform the full source mesh
    aligned_vertices = (source_vertices - src_mean) @ R * scale + tgt_mean

    return aligned_vertices

def export_ply(ply, path, array,
               distances = None,
               colorize: bool = False,
               vmin: float = 0.0,
               vmax: float = 1.0,
               cmap_name: str = 'viridis'):
    """
    Export a PLY, optionally coloring its vertices by a distance array.
    
    """
    
    ply["vertex"]["x"] = array[:,0]
    ply["vertex"]["y"] = array[:,1]
    ply["vertex"]["z"] = array[:,2]
    
    if not colorize:
        # classic path: ply is assumed to be a dict-like PlyData.struct
        modified = PlyData(ply, text=False)
        modified.write(path)
        return

    # colorize == True
    if distances is None:
        raise ValueError("Must supply `distances` when colorize=True")
    # Load existing PLY from file
    plydata = ply
    vdata = plydata['vertex'].data  # structured array

    # Normalize distances into [0,1]
    norm = np.clip((distances - vmin) / (vmax - vmin), 0.0, 1.0)
    # Map through colormap → (N,4), drop alpha
    cmap = cm.get_cmap(cmap_name)
    rgbf = cmap(norm)[:, :3]
    rgb = (rgbf * 255).astype(np.uint8)

    # Build new structured array
    old_descr = vdata.dtype.descr
    new_descr = old_descr + [('red','u1'), ('green','u1'), ('blue','u1')]
    new_vdata = np.empty(vdata.shape, dtype=new_descr)
    # copy old fields & assign colors
    for name in vdata.dtype.names:
        new_vdata[name] = vdata[name]
    new_vdata['red']   = rgb[:, 0]
    new_vdata['green'] = rgb[:, 1]
    new_vdata['blue']  = rgb[:, 2]

    # Describe and write out
    new_vertex = PlyElement.describe(new_vdata, 'vertex')
    others = [e for e in plydata.elements if e.name != 'vertex']
    out = PlyData([new_vertex] + others, text=False)
    out.write(path)


def compute_vertex_to_surface_distance(source_vertices, target_mesh_path,lm_target):
    """
    Compute the closest surface distance from each vertex of the source mesh
    to the surface of the target mesh.

    """
    
    # Derive PLY path from STL path
    path_ply = os.path.splitext(target_mesh_path)[0] + '.ply'

    # Try loading STL first
    try:
        mesh = trimesh.load(target_mesh_path, force='mesh', process=False)
        print("Loaded STL successfully.")
    except Exception as e:
        print(f"Failed to load STL: {e}")
        print("Trying to load PLY instead...")
        try:
            mesh = trimesh.load(path_ply, force='mesh', process=False)
            print("Loaded PLY successfully.")
        except Exception as e2:
            print(f"Failed to load PLY as well: {e2}")
            mesh = None
            return [0],[0]
    
    # Compute closest points and distances
    diff = mesh.vertices - lm_target[0]  # shape: (N, 3)
    dists = np.linalg.norm(diff, axis=1)  # Euclidean distance to the landmark
    min_dist = np.min(dists)
    
    if min_dist<10:
        closest_points, distances, _ = mesh.nearest.on_surface(source_vertices)
        return distances, closest_points
    else:
        print("landmarks do not match mesh vertices")
        return [0],[0]
