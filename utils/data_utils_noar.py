import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
import networkx as nx

import torch#, tqdm
import tqdm as tqdm_outer
from tqdm import tqdm
import torch.nn.functional as F
# from multiprocessing import Pool
import torch.multiprocessing as mp
# import dgl.multiprocessing as mp
# import multiprocessing as mp
#mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
#mp.set_sharing_strategy('file_system')
import glob, pickle, random
import os
import os.path as osp
import copy
# from torch_geometric.data import Dataset, DataLoader
# from torch_geometric.transforms import BaseTransform
from collections import defaultdict
from molecule_utils import *
# from torch_scatter import scatter
# from torch_geometric.data import Dataset, Data, DataLoader
from dgl.data import DGLDataset
from dgl.dataloading import DataLoader
# import psutil
import concurrent.futures
from rdkit.Geometry import Point3D
import time

def calculate_dihedral_angle(pos, iAtomId, jAtomId, kAtomId):
#     points = [Point3D(x, y, z) for x, y, z in positions]
    rJI = (pos[iAtomId] - pos[jAtomId])
    rJI = Point3D(rJI[0], rJI[1], rJI[2])
    rJK = (pos[kAtomId] - pos[jAtomId])
    rJK = Point3D(rJK[0], rJK[1], rJK[2])
    
    return rJI.AngleTo(rJK)

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}
QM9_DIMS = ([5, 4, 2, 8, 6, 8, 4, 6, 5], 0)
DRUG_DIMS = ([35, 4, 2, 8, 6, 8, 4, 6, 5], 0)
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
def generated_rdkit_confs_no_mmff(mol, count, seed = None, use_mmff = False, threads = 16):
    mol = copy.deepcopy(mol)
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=count, numThreads=threads)
    indices = list(range(mol.GetNumConformers()))
    if mol is None or cids is None or len(cids) == 0:
        print(f"Mol or cid or energies are None")
        return None
    elif len(indices) < count:
        for _ in range(count - len(indices)):
            first_conformer = mol.GetConformer(random.choice(indices))
            new_conformer = Chem.Conformer(len(first_conformer.GetPositions()))
            for i, coord in enumerate(first_conformer.GetPositions()):
                new_conformer.SetAtomPosition(i, coord)
            # mol.AddConformer(new_conformer)
            mol.AddConformer(new_conformer, assignId= True) # for QM9
        indices = list(range(count))
    if mol.GetNumConformers() < count or len(indices) != count:
        import ipdb; ipdb.set_trace()
        test= 1
    assert(mol.GetNumConformers() >= count and len(indices) == count)
    rd_coords = []
    for idx in indices:
        try:
            rd_coords.append(np.asarray(mol.GetConformer(idx).GetPositions()))
        except Exception as e:
            print(e)
            print("weird qm9 bug")
            import ipdb; ipdb.set_trace()
            test = 1
            return None
    return rd_coords
    
def generated_rdkit_confs(mol, count, seed = None, use_mmff = True, threads = 16):
    # import ipdb; ipdb.set_trace()
    mol = copy.deepcopy(mol)
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=3*count, numThreads=threads)
    try:
        # AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s', numThreads = threads)
    except:
        print(f"Could not use mmff")
        return None
    info = (cids, res)
    if mol is None or cids is None or len(cids) == 0 or info is None:
        print(f"Mol or cid or energies are None")
        return None
    indices = [idx for idx, val in enumerate(info[1]) if val[0] == 0] # check confs that converged
    if len(indices) == 0:
        print(f"MMFF did not converge so just EKDTG")
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=count, numThreads=threads)
        if len(cids) < count:
            return None
        indices = list(range(count))
    elif len(indices) > count:
        energies = [(idx, val[1]) for idx, val in enumerate(info[1]) if val[0] == 0]
        energies.sort(key=lambda x: x[1])
        ids = [x[0] for x in energies[:count]]
        indices = [idx for idx in indices if idx in ids]
    elif len(indices) < count:
        for _ in range(count - len(indices)):
            first_conformer = mol.GetConformer(random.choice(indices))
            new_conformer = Chem.Conformer(len(first_conformer.GetPositions()))
            for i, coord in enumerate(first_conformer.GetPositions()):
                new_conformer.SetAtomPosition(i, coord)
            # mol.AddConformer(new_conformer)
            mol.AddConformer(new_conformer, assignId= True) # for QM9
        indices = list(range(count))
    # 
    if mol.GetNumConformers() < count or len(indices) != count:
        import ipdb; ipdb.set_trace()
        test= 1
    assert(mol.GetNumConformers() >= count and len(indices) == count)
    rd_coords = []
    for idx in indices:
        try:
            rd_coords.append(np.asarray(mol.GetConformer(idx).GetPositions()))
        except Exception as e:
            print(e)
            print("weird qm9 bug")
            import ipdb; ipdb.set_trace()
            test = 1
            return None
    return rd_coords

def get_angle_graph(mol, torsions):
    torsion_angles = {}
    for a,b,c,d in torsions:
        torsion_angles[(a,b,c,d)] = rdMolTransforms.GetDihedralRad(mol.GetConformer(), a, b, c, d)
        # if torsion_angles[(a,b,c,d)] < 0: #! Matches
        #     pos = mol.GetConformer().GetPositions()
        #     import ipdb; ipdb.set_trace()
        #     p0, p1, p2, p3 = pos[a], pos[b], pos[c], pos[d]
        #     s1 = torch.tensor(p1 - p0)
        #     s2 = torch.tensor(p2 - p1)
        #     s3 = torch.tensor(p3 - p2)
        #     sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
        #     cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)
        #     angle = torch.atan2(sin_d_, cos_d_ + 1e-10)
        #     test = angle
    dihedral_angles = {}
    # Iterate through the atoms in the molecule
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        # Get the neighbors of the current atom
        neighbors = atom.GetNeighbors()
        # Iterate through the neighbors
        for neighbor in neighbors:
            neighbor_idx = neighbor.GetIdx()
            # Check if the current atom and its neighbor are bonded
            if mol.GetBondBetweenAtoms(atom_idx, neighbor_idx) is not None:
                # Get the bonded neighbor of the current neighbor
                bonded_neighbor_indices = [n.GetIdx() for n in neighbor.GetNeighbors()]
                for bonded_neighbor_idx in bonded_neighbor_indices:
                    if bonded_neighbor_idx != atom_idx:
                        # Calculate the dihedral angle between the atoms
                        angleA = rdMolTransforms.GetAngleRad(
                            mol.GetConformer(), atom_idx, neighbor_idx, bonded_neighbor_idx
                        )
                        # angleB = calc_dihedral(
                        #     torch.tensor(mol.GetConformer().GetPositions()),  (atom_idx, neighbor_idx, bonded_neighbor_idx)
                        # ).item()
                        # angleC = calculate_dihedral_angle(mol.GetConformer().GetPositions(),  atom_idx, neighbor_idx, bonded_neighbor_idx)
                        # print(angleA, angleB, angleC)
                        # import ipdb; ipdb.set_trace()
                        dihedral_angles[(atom_idx, neighbor_idx, bonded_neighbor_idx)] = angleA
                        # angleB = calc_dihedral(
                        #     torch.tensor(mol.GetConformer().GetPositions()),  (atom_idx, neighbor_idx, bonded_neighbor_idx)
                        # )
    return (torsion_angles, dihedral_angles)

def calc_dihedral(pos, k): #a, b):
        """
        Compute angle between two batches of input vectors in radians
        """
        # TODO make this work for batches
        a,b,c = k
        a_pos = pos[a]
        b_pos = pos[b]
        c_pos = pos[c]
    
        AB = a_pos - b_pos
        CB = c_pos - b_pos
        inner_product = (AB * CB).sum(dim=-1)
        # norms
        ABn = torch.linalg.norm(AB, dim=-1)
        BCn = torch.linalg.norm(AB, dim=-1)
        # protect denominator during division
        den = ABn * BCn + 1e-10
        den2 = torch.sqrt((AB*AB).sum(-1)*(CB*CB).sum(-1)) + 1e-10
        cos = inner_product / den2
        cos = torch.clamp(cos, min=-1.0, max=1.0) # some values are -1.005
        if torch.any(torch.isnan(cos)) or torch.any(torch.isnan(torch.acos(cos))):
            import ipdb; ipdb.set_trace()
            test= 1
        # Calculate the angle in radians using arccosine
        # return cos
        return torch.acos(cos)
    
def check_distances(molecule, geometry_graph, B = False):
    src, dst = geometry_graph.edges()
    src = src.long()
    dst = dst.long()
    generated_coords = molecule.ndata['x'] #!!! THIS WAS THE BUG X_TRUE DOES NOT MAKE SENSE FOR RDKIT
    d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
    error = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
    return error

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding

def get_bond_idx(m, i, j):
    bond = m.GetBondBetweenAtoms(i,j)
    if bond == None:
        return 4 #-1e9 # not inf for stab
    else:
        return bonds[bond.GetBondType()]

def featurize_molecule(mol, types=drugs_types, rdkit_coords = None, seed = 0, radius = 4, max_neighbors=None):
    if type(types) is str:
        if types == 'qm9':
            types = qm9_types
        elif types == 'drugs':
            types = drugs_types
    
    N = mol.GetNumAtoms()
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(types[atom.GetSymbol()])
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        
        atom_features.extend([atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))#6
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))#8
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))#4
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])#6
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3])) # 5

#     z = torch.tensor(atomic_number, dtype=torch.long)

#     row, col, edge_type = [], [], []
#     for bond in mol.GetBonds():
#         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_type += 2 * [bonds[bond.GetBondType()]]

#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     edge_type = torch.tensor(edge_type, dtype=torch.long)
#     edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(types)) # 5
    x2 = torch.tensor(atom_features).view(N, -1) # 39
    x3 = torch.tensor(chiral_tag).view(N, -1).to(torch.float) # 1
    node_features = torch.cat([x1.to(torch.float), x3, x2], dim=-1)
    
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    # import ipdb; ipdb.set_trace()
#     use_rdkit_coords = True
    if rdkit_coords is not None:
        R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
        lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    remove_centroid = True
    if remove_centroid:
        lig_coords -= np.mean(lig_coords, axis = 0)
        true_lig_coords -= np.mean(true_lig_coords, axis = 0)
        
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    bond_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            #print( f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        assert dst != []
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
#         bonds = []
#         for d in dst:
#             bonds.append(get_bond(mol, int(i), int(d)))
#         bond_list.extend(bonds)
        bond_list.extend([get_bond_idx(mol, int(i), int(d)) for d in dst])
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = node_features
    graph.ndata['ref_feat'] = node_features
    # import ipdb; ipdb.set_trace()
    edge_type = torch.from_numpy(np.asarray(bond_list).astype(np.float32)).type(torch.long) #torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)
    graph.edata['feat'] = torch.cat((distance_featurizer(dist_list, 0.75), edge_attr), dim = -1)

    graph.ndata['x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_ref'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_true'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
        
    return G
def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list

def get_transformation_mask(mol, pyg_data = None):
#     G = to_networkx(pyg_data, to_undirected=False)
    G = mol_to_nx(mol)
    tas = get_torsion_angles(mol)
    tas2 = get_torsions_geo([mol])
    to_rotate = []
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
    # import ipdb; ipdb.set_trace()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edges = edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])
    # import ipdb; ipdb.set_trace()
    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1
    # import ipdb; ipdb.set_trace()
    return mask_edges, mask_rotate

def get_transformation_mask2(mol, pyg_data = None, angles = None, mask_check = None):
#     G = to_networkx(pyg_data, to_undirected=False)
    G = mol_to_nx(mol)
    # tas = get_torsion_angles(mol)
    if angles is None:
        tas2 = get_torsions_geo([mol])
    else:
        tas2 = angles
    torsion_map = {}
    for b in tas2:
        torsion_map[(b[1], b[2])] = ([b[0], b[3]])
    U, V = pyg_data.edges()
    to_rotate = []
    row, col, edge_type = [], [], []
    # mask_edges2 = []
    # import ipdb; ipdb.set_trace()
    seen = set()
    for start, end in zip(U,V):
        start = start.item()
        end = end.item()
        if (start, end) in seen:
            continue
        seen.add((end, start))
        row += [start, end]
        col += [end, start]
        # mask_edges2 += [(start, end) in torsion_map, (end, start) in torsion_map]
    # for bond in mol.GetBonds():
    #     start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #     row += [start, end]
    #     col += [end, start]
    #     edge_type += 2 * [bonds[bond.GetBondType()]]
    # import ipdb; ipdb.set_trace()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edges = edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]
        if (tuple(edges[i]) in torsion_map or tuple(edges[i][::-1]) in torsion_map):
            G2 = G.to_undirected()
            if G2.has_edge(*edges[i]):
                G2.remove_edge(*edges[i])
                if not nx.is_connected(G2):
                    l = list(sorted(nx.connected_components(G2), key=len)[0])
                    if len(l) > 1:
                        if edges[i, 0] in l:
                            to_rotate.append([])
                            to_rotate.append(l)
                        else:
                            to_rotate.append(l)
                            to_rotate.append([])
                        continue
        to_rotate.append([])
        to_rotate.append([])
    # import ipdb; ipdb.set_trace()
    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    # mask_edges2 = np.asarray(mask_edges2, dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    # if mask_check is not None:
    #     if mask_rotate.shape != mask_check:
    #         print("angle mask issue")
    #         import ipdb; ipdb.set_trace()
    #         test = 1
    idx = 0
    for i in range(mask_edges.shape[0]):#range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1
    # import ipdb; ipdb.set_trace()
    return mask_edges, mask_rotate

class ConformerDataset(DGLDataset):
    def __init__(self, root, split_path, mode, types, dataset, num_workers=1, limit_molecules=None,
                 cache_path=None, pickle_dir=None, boltzmann_resampler=None, raw_dir='/home/dannyreidenbach/data/QM9/dgl', save_dir='/home/dannyreidenbach/data/QM9/dgl',
                 force_reload=False, verbose=False, transform=None, name = "qm9",
                 invariant_latent_dim = 64, equivariant_latent_dim = 32, use_diffusion_angle_def = False, old_rdkit = False):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol
#         super(ConformerDataset, self).__init__(name, transform) /data/dreidenbach/data
        self.D = invariant_latent_dim
        self.F = equivariant_latent_dim
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.boltzmann_resampler = boltzmann_resampler
        self.cache_path = cache_path
        self.use_diffusion_angle_def = use_diffusion_angle_def
        # print("Cache", cache_path)
        # if cache_path: cache_path += "." + mode
        self.use_name = name
        self.split_path = split_path
        self.mode = mode
        self.pickle_dir = pickle_dir
        self.num_workers = num_workers
        self.limit_molecules = limit_molecules
        self.old_rdkit = old_rdkit
        super(ConformerDataset, self).__init__(name, raw_dir = raw_dir, save_dir = save_dir, transform = transform)
        
    def process(self):
        if self.cache_path and os.path.exists(self.cache_path):
            print('Reusing preprocessing from cache', self.cache_path)
            with open(self.cache_path, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            # if self.dataset == 'qm9':
            #     #  self.datapoints = self.preprocess_datapoints(self.root, self.split_path, self.pickle_dir, self.mode, self.num_workers, self.limit_molecules)
            #     raise ValueError("QM9 processing not supported yet")
            # import ipdb; ipdb.set_trace()
            self.datapoints = self.preprocess_datapoints_chunk(self.root, self.split_path, self.pickle_dir, self.mode, self.num_workers, self.limit_molecules)
            # import ipdb; ipdb.set_trace()
            if self.cache_path:
                print("Caching at", self.cache_path)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.datapoints, f)
            
    def preprocess_datapoints_chunk(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]
        self.open_pickles = {}
        smiles = [smi[len(root):-7] for smi in smiles]
        print('Preparing to process', len(smiles), 'smiles')
        chunk_size = len(smiles)//5
        all_smiles = smiles
        smiles = []
        old_name = self.use_name
        total_count = 0
        for i in range(6):
            datapoints = []
            # if i > 0:
            #     import ipdb; ipdb.set_trace()
            smiles = all_smiles[i*chunk_size : (i+1)*chunk_size]
            chunk_amount = len(smiles)
            if num_workers > 1:
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    with tqdm(total=len(smiles)) as pbar:
                        futures = [executor.submit(self.filter_smiles_mp, (chunk_amount, entry)) for entry in enumerate(smiles)]
                        for future in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                    molecules = [future.result() for future in concurrent.futures.as_completed(futures)]
                    datapoints.extend([item for sublist in molecules for item in sublist if sublist is not None and sublist[0] is not None])
                    
                # molecules = []
                # with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                #     with tqdm(total=len(smiles)) as pbar:
                #         futures = [executor.submit(self.filter_smiles_mp, entry) for entry in smiles]
                #         for future in concurrent.futures.as_completed(futures):
                #             try:
                #                 result = future.result(timeout=1200)  # 20 minutes timeout in seconds
                #             except concurrent.futures.TimeoutError:
                #                 result = None  # Handle timeout
                #             pbar.update(1)
                #             molecules.append(result)
                #         datapoints.extend([item for sublist in molecules for item in sublist if sublist is not None and sublist[0] is not None])
            else:
                chunk_amount = len(smiles)
                molecules = []
                for entry in enumerate(smiles):
                    molecules.append(self.filter_smiles_mp((chunk_amount, entry)))
                    import ipdb; ipdb.set_trace()
                datapoints.extend([item for sublist in molecules for item in sublist if sublist is not None and sublist[0] is not None])
            print('Fetched', len(datapoints), 'mols successfully')
            total_count += len(datapoints)
            print('Fetched Total', total_count, 'mols successfully')
            print(self.failures)
            if pickle_dir: del self.current_pickle
            self.datapoints = datapoints
            self.use_name = old_name + f"_{i}"
            self.save()
            # import ipdb; ipdb.set_trace()
            test = 1
        # import ipdb; ipdb.set_trace()
        test = 1
        return datapoints
        
    def filter_smiles_mp(self, smile):
        chunk_amount, smile = smile
        idx, smile = smile
        # if idx +1 ==chunk_amount:
        #     import ipdb; ipdb.set_trace()
        #     test=1
        # if idx > 48675:
        #     import ipdb; ipdb.set_trace()
        #     test=1
        # else:
        #     return [None]
        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    # print("A")
                    return [None]
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                # print("B")
                return [None]
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
                self.failures['raw_pickle_not_found'] += 1
                # print("C")
                return [None]
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)

        smile = mol_dic['smiles']
        # print(smile)
        if '.' in smile:
            self.failures['dot_in_smile'] += 1
            # print("D")
            return [None]
        # import ipdb; ipdb.set_trace()
        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            # print("E")
            return [None]

        mol = mol_dic['conformers'][0]['rd_mol']
        # xc = mol.GetConformer().GetPositions()
        # print("filter mol POS", mol.GetConformer().GetPositions())
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            # print("F")
            return [None]

        if N < 4:
            self.failures['mol_too_small'] += 1
            # print("G")
            return [None]
        # print("A filter")
        # import ipdb; ipdb.set_trace()
        datas, data_Bs = self.featurize_mol(mol_dic)
        
        if not datas or len(datas) == 0 or not data_Bs or len(data_Bs) == 0:
            self.failures['featurize_mol_failed'] += 1
            # print("H")
            return [None]
        
        
        results_A = []
        results_B = []
        bad_idx_A, bad_idx_B = [], []
        angles = get_torsions_geo([mol])
        angle_masks_A = []
        mask_check = None
        for idx, (midx, data) in enumerate(datas):
            if not data:
                self.failures['featurize_mol_failed_A'] += 1
                bad_idx_A.append(idx)
                results_A.append(None)
                continue
            mol = mol_dic['conformers'][midx]['rd_mol']
            # import ipdb; ipdb.set_trace()
            edge_mask, mask_rotate = get_transformation_mask2(mol, data, angles, mask_check)
            # if mask_check is None:
            #     mask_check = mask_rotate.shape
            mask_rotate_ten = torch.tensor(mask_rotate).T
            edge_mask_ten = torch.tensor(edge_mask)
            if np.sum(edge_mask) < 0.5: #TODO: Do we need this since we are using GEOMOL
                self.failures['no_rotable_bonds'] += 1
                bad_idx_A.append(idx)
                results_A.append(None)
                continue
            # print("filter SMILE", smile)
            # import ipdb; ipdb.set_trace()
            # data.ndata['mask_rotate_T'] = mask_rotate_ten
            # angle_masks_A.append(mask_rotate_ten)
            angle_masks_A = mask_rotate_ten
            data.edata['mask_edges'] = edge_mask_ten
            try:
                # import ipdb; ipdb.set_trace()
                torsions_A, A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule_new(mol)
                A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
            except Exception as e:
                print(e)
                self.failures['coarsening error'] += 1
                print(f"coarsening failure", mol_dic['smiles'])
                bad_idx_A.append(idx)
                results_A.append(None)
                continue
            # xcc = mol.GetConformer().GetPositions()
            # print("filter mol POS2", xcc == xc)
            geometry_graph_A = get_geometry_graph(mol)
            # import ipdb; ipdb.set_trace()
            angle_graph_A = get_angle_graph(mol, torsions_A) #TODO: Implement this graph
            err = check_distances(data, geometry_graph_A)
            assert(err < 1e-3)
            Ap = create_pooling_graph(data, A_frag_ids)
            geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
            results_A.append((data, geometry_graph_A, angle_graph_A, angle_masks_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))

        # for idx, data_B in enumerate(data_Bs):
        mask_check = None
        angle_masks_B = []
        for idx, (midx, data_B) in enumerate(data_Bs):
            if idx in set(bad_idx_A):
                bad_idx_B.append(idx)
                results_B.append(None)
                continue
            if not data_B:
                self.failures['featurize_mol_failed_B'] += 1
                bad_idx_B.append(idx)
                # print("BAD B", idx, len(data_Bs))
                results_B.append(None)
                continue
                # return [None]
            mol = mol_dic['conformers'][midx]['rd_mol']
            edge_mask, mask_rotate = get_transformation_mask2(mol, data_B, angles, mask_check)
            # if mask_check is None:
            #     mask_check = mask_rotate.shape
            mask_rotate_ten = torch.tensor(mask_rotate).T
            edge_mask_ten = torch.tensor(edge_mask)
            torsions_B, B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule_new(mol)
            B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
#             geometry_graph_B = copy.deepcopy(geometry_graph_A) #get_geometry_graph(mol)
            geometry_graph_B = get_geometry_graph(mol, data_B.ndata['x'].numpy())
            Bp = create_pooling_graph(data_B, B_frag_ids)
            geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
            err = check_distances(data_B, geometry_graph_B, True)
            angle_graph_B = get_angle_graph(mol, torsions_B) # this is same as other since mol uses GT as its conformer for RDKit
            # data_B.ndata['mask_rotate_T'] =  #! Cannot put in dgl
            # angle_masks_B.append(mask_rotate_ten)
            angle_masks_B = mask_rotate_ten
            data_B.edata['mask_edges'] = edge_mask_ten
            # import ipdb; ipdb.set_trace()
            # if err >= 1e-3:
            #     import ipdb; ipdb.set_trace()
            #     geometry_graph_B = get_geometry_graph(mol, data_B.ndata['x'].numpy())
            assert(err < 1e-3)
            results_B.append((data_B, geometry_graph_B, angle_graph_B, angle_masks_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
        assert(len(results_A) == len(results_B))
        bad_idx = set(bad_idx_A) | set(bad_idx_B)
        results_A = [x for idx, x in enumerate(results_A) if idx not in bad_idx]
        results_B = [x for idx, x in enumerate(results_B) if idx not in bad_idx]
        assert(len(results_A) == len(results_B))
        if len(results_A) == 0 or len(results_B) == 0:
            # print("Bad Input")
            # print("I")
            return [None]
        # print(smile, len(results_A))
        return [(a,b) for a,b in zip(results_A, results_B)]

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        return copy.deepcopy(data)
    
    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        data = self.datapoints[idx]
        return copy.deepcopy(data)

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.datapoints)
    
    def save(self):
        # return True
        graphs, infos = [], []
        for A, B in self.datapoints:
            data_A, geometry_graph_A, angle_graph_A, angle_masks_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids = A
            data_B, geometry_graph_B, angle_graph_B, angle_masks_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids = B
            infos.append((A_frag_ids, B_frag_ids, angle_graph_A, angle_graph_B, angle_masks_A, angle_masks_B))
            graphs.extend([data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg])
        dgl.data.utils.save_info(self.save_dir + f'/{self.use_name}_infos_no_ar.bin', infos)
        dgl.data.utils.save_graphs(self.save_dir + f'/{self.use_name}_graphs_no_ar.bin', graphs)
        print("Saved Successfully", self.save_dir, self.use_name, len(self.datapoints))
    
    def load(self):
        if False: #self.dataset == "qm9" or self.dataset == 'xl':
            # graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_graphs_no_ar.bin')
            # info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_infos_no_ar.bin')
            # count = 0
            # results_A, results_B = [], []
            # for i in range(0, len(graphs), 10):
            #     AB = graphs[i: i+10]
            #     A_frag_ids, B_frag_ids = info[count]
            #     count += 1
            #     data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
            #     data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
            #     results_A.append((data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
            #     results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
            # self.datapoints = [(a,b) for a,b in zip(results_A, results_B)]
            # print("Loaded Successfully",  self.save_dir, self.use_name, len(self.datapoints))
            raise ValueError("QM9 not yet supported")
        else:
            if False:
                try:
                    count = 0
                    results_A, results_B = [], []
                    self.datapoints = []
                    print(f"Loading Data ...")
                    graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_graphs_no_ar.bin')
                    info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_infos_no_ar.bin')
                    results_A, results_B = [], []
                    for i in range(0, len(graphs), 10):
                        AB = graphs[i: i+10]
                        A_frag_ids, B_frag_ids, angle_graph_A, angle_graph_B = info[count]
                        count += 1
                        data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
                        data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
                        results_A.append((data_A, geometry_graph_A, angle_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                        results_B.append((data_B, geometry_graph_B, angle_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
                    self.datapoints.extend([(a,b) for a,b in zip(results_A, results_B)])
                    print("Loaded Successfully",  self.save_dir, self.use_name, len(self.datapoints))
                except:
                    import ipdb; ipdb.set_trace()
            else:
                try:
                    count = 0
                    results_A, results_B = [], []
                    self.datapoints = []
                    for chunk in range(6): #! FOR DRUGS use 6 else 5
                        if chunk < 0: #! No Skip
                            print("Skipping large chunks for debugging", chunk)
                            break
                        start_time = time.time()
                        print(f"Loading Chunk {chunk} ...")
                        # graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_{chunk}_graphs_no_ar_test2.bin')
                        # info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_{chunk}_infos_no_ar_test2.bin')
                        graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_{chunk}_graphs_no_ar.bin')
                        info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_{chunk}_infos_no_ar.bin')
                        count = 0
                        print(f"Loading Chunk {chunk} = {len(graphs)//10}")
                        results_A, results_B = [], []
                        for i in range(0, len(graphs), 10):
                            AB = graphs[i: i+10]
                            A_frag_ids, B_frag_ids, angle_graph_A, angle_graph_B, angle_masks_A, angle_masks_B = info[count]
                            # A_frag_ids, B_frag_ids, angle_graph_A, angle_graph_B = info[count]
                            # angle_masks_A, angle_masks_B = [], []
                            count += 1
                            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
                            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
                            # del data_A.ndata['mask_rotate_T']
                            # del data_B.ndata['mask_rotate_T']
                            results_A.append((data_A, geometry_graph_A, angle_graph_A, angle_masks_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                            results_B.append((data_B, geometry_graph_B, angle_graph_B, angle_masks_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
                        self.datapoints.extend([(a,b) for a,b in zip(results_A, results_B)])
                        print(f"Loaded Chunk {chunk}", time.time() - start_time)
                    print("Loaded Successfully",  self.save_dir, self.use_name, len(self.datapoints))
                except:
                    import ipdb; ipdb.set_trace()
                
    
    def has_cache(self):
        # if self.dataset == "qm9":
        #     return os.path.exists(os.path.join(self.save_dir, f'{self.use_name}_graphs.bin'))
        # else:
        return os.path.exists(os.path.join(self.save_dir, f'{self.use_name}_0_graphs_no_ar.bin')) #_0_graphs_no_ar
    #  return os.path.exists(os.path.join(self.save_dir, f'{self.use_name}_0_graphs_no_ar_test2.bin')) #_0_graphs_no_ar

    def __repr__(self):
        return f'Dataset("{self.name}", num_graphs={len(self)},' + \
               f' save_path={self.save_path})'

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic, limit = 5):
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        datas = []
        # nconfs = len(confs[:nlimit])
        for idx, conf in enumerate(confs[:limit]):
            mol = Chem.AddHs(conf['rd_mol'])
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue
            if conf_canonical_smi != canonical_smi:
                continue
            datas.append((idx, mol))
        # return None if no non-reactive conformers were found
        if len(datas) == 0:
            return None, None
        # import ipdb; ipdb.set_trace()
        mol = datas[0][-1]
        rd_coords = generated_rdkit_confs(mol, len(datas))
        if rd_coords == None:
            return None, None
        try:
            gt_datas = [(correct_mol[0], featurize_molecule(correct_mol[1], self.types, rdkit_coords = None)) for idx, correct_mol in enumerate(datas)]
            rd_datas = [(correct_mol[0], featurize_molecule(correct_mol[1], self.types, rdkit_coords = rd_coords[idx])) for idx, correct_mol in enumerate(datas)]
        except Exception as e:
            print(e)
            # import ipdb; ipdb.set_trace()
            test = 1
            gt_datas = None
            rd_datas = None
        return gt_datas, rd_datas

    # def resample_all(self, resampler, temperature=None):
    #     ess = []
    #     for data in tqdm.tqdm(self.datapoints):
    #         ess.append(resampler.resample(data, temperature=temperature))
    #     return ess

def collate(samples):
    A, B = map(list, zip(*samples))
#     data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg
#  .to('cuda:0') causes errors
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    angle_A = [x[2] for x in A]
    angle_masks_A = [x[3] for x in A]
    Ap = dgl.batch([x[4] for x in A])
    A_cg = dgl.batch([x[5] for x in A])
    geo_A_cg = dgl.batch([x[6] for x in A])
    frag_ids = [x[7] for x in A]
    
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    angle_B = [x[2] for x in B]
    angle_masks_B = [x[3] for x in B]
    Bp = dgl.batch([x[4] for x in B])
    B_cg = dgl.batch([x[5] for x in B])
    geo_B_cg = dgl.batch([x[6] for x in B])
    B_frag_ids = [x[7] for x in B]
    return (A_graph, geo_A, angle_A, angle_masks_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, angle_B, angle_masks_B, Bp, B_cg, geo_B_cg, B_frag_ids)


def cook_drugs_angles(batch_size = 32, mode = 'train', data_dir='/data/dreidenbach/data/torsional_diffusion/DRUGS/drugs/',
                dataset='drugs', limit_mols=0, log_dir='./test_run', num_workers=20, restart_dir=None, seed=0,
                 split_path='/data/dreidenbach/data/torsional_diffusion/DRUGS/split.npy',
                 std_pickles=None):
    types = drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_no_ar_final_all', #! remove all when going to 100 limit test
                                   pickle_dir=std_pickles,
                                   raw_dir='/data/dreidenbach/data/DRUGS/dgl', 
                                   save_dir='/data/dreidenbach/data/DRUGS/dgl',
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   boltzmann_resampler=None)
    # import ipdb; ipdb.set_trace()
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
                                            collate_fn = collate)
    return dataloader, data

def cook_drugs_ot(batch_size = 32, mode = 'train', data_dir='/data/dreidenbach/data/torsional_diffusion/DRUGS/drugs/',
                dataset='drugs', limit_mols=0, log_dir='./test_run', num_workers=20, restart_dir=None, seed=0,
                 split_path='/data/dreidenbach/data/torsional_diffusion/DRUGS/split.npy',
                 std_pickles=None):
    types = drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=0,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_no_ar_final_all', #! remove all when going to 100 limit test
                                   pickle_dir=std_pickles,
                                   raw_dir='/data/dreidenbach/data/DRUGS/dgl', 
                                   save_dir='/data/dreidenbach/data/DRUGS/dgl',
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   boltzmann_resampler=None)
    # import ipdb; ipdb.set_trace()
    limit = 5
    batch_size = batch_size//limit
    mdata = []
    chunk = []
    counts = []
    size = -1
    angles = None
    count = 0
    all_count = 0
    # import ipdb; ipdb.set_trace()
    for graphs in data.datapoints:
        # if all_count == 1000:
        #     import ipdb; ipdb.set_trace()
        # all_count += 1
        A, B = graphs
        if size == -1:
            size =  A[0].ndata['x'].shape[0]
            angles = A[2][0].keys()
        if len(chunk) >= limit:
            size =  A[0].ndata['x'].shape[0]
            angles = A[2][0].keys()
            if len(chunk) > 0:
                mdata.append(chunk)
            chunk = []
            
        if size == A[0].ndata['x'].shape[0] and angles == A[2][0].keys():
            chunk.append(graphs)
        else:
            size =  A[0].ndata['x'].shape[0]
            angles = A[2][0].keys()
            if len(chunk) > 0:
                mdata.append(chunk)
            chunk = []
    # import ipdb; ipdb.set_trace()
    data.datapoints = mdata
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, collate_fn = collate_mols)
    # import ipdb; ipdb.set_trace()
    return dataloader, mdata

def collate_mols(samples):
    # import ipdb; ipdb.set_trace()
    # A, B = map(list, zip(*samples))
    A, B = zip(*[(t[0], t[1]) for sublist in samples for t in sublist])
#     data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg
#  .to('cuda:0') causes errors
    nmols = [len(x) for x in samples]

    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    angle_A = [x[2] for x in A]
    angle_masks_A = [x[3] for x in A]
    Ap = dgl.batch([x[4] for x in A])
    A_cg = dgl.batch([x[5] for x in A])
    geo_A_cg = dgl.batch([x[6] for x in A])
    frag_ids = [x[7] for x in A]
    
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    angle_B = [x[2] for x in B]
    angle_masks_B = [x[3] for x in B]
    Bp = dgl.batch([x[4] for x in B])
    B_cg = dgl.batch([x[5] for x in B])
    geo_B_cg = dgl.batch([x[6] for x in B])
    B_frag_ids = [x[7] for x in B]
    return (A_graph, geo_A, angle_A, angle_masks_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, angle_B, angle_masks_B, Bp, B_cg, geo_B_cg, B_frag_ids), nmols

def cook_qm9_ot(batch_size = 32, mode = 'train', data_dir='/data/dreidenbach/data/torsional_diffusion/QM9/qm9/',
                dataset='qm9', limit_mols=0, log_dir='./test_run', num_workers=20, restart_dir=None, seed=0,
                 split_path='/data/dreidenbach/data/torsional_diffusion/QM9/split.npy',
                 std_pickles=None):
    types = qm9_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=10,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_no_ar_final_all', #! remove all when going to 100 limit test
                                   pickle_dir=std_pickles,
                                   raw_dir='/data/dreidenbach/data/QM9/dgl', 
                                   save_dir='/data/dreidenbach/data/QM9/dgl',
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   boltzmann_resampler=None)
    # import ipdb; ipdb.set_trace()
    limit = 5
    batch_size = batch_size//limit
    mdata = []
    chunk = []
    counts = []
    size = -1
    angles = None
    count = 0
    all_count = 0
    # import ipdb; ipdb.set_trace()
    for graphs in data.datapoints:
        # if all_count == 1000:
        #     import ipdb; ipdb.set_trace()
        # all_count += 1
        A, B = graphs
        if size == -1:
            size =  A[0].ndata['x'].shape[0]
            angles = A[2][0].keys()
        if len(chunk) >= limit:
            size =  A[0].ndata['x'].shape[0]
            angles = A[2][0].keys()
            if len(chunk) > 0:
                mdata.append(chunk)
            chunk = []
            
        if size == A[0].ndata['x'].shape[0] and angles == A[2][0].keys():
            chunk.append(graphs)
        else:
            size =  A[0].ndata['x'].shape[0]
            angles = A[2][0].keys()
            if len(chunk) > 0:
                mdata.append(chunk)
            chunk = []
    # import ipdb; ipdb.set_trace()
    data.datapoints = mdata
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, collate_fn = collate_mols)
    # import ipdb; ipdb.set_trace()
    return dataloader, mdata

