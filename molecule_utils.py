import os
import sys
import pickle
import numpy as np
import dgl
from collections import defaultdict, Counter

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import torch
from geometry_utils import *
import scipy.spatial as spa
from scipy import spatial
from scipy.special import softmax
from embedding import *

def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def align_coordinates(rd_mol, kit_mol):
    prior_coords = kit_mol.GetConformer().GetPositions()
    true_coords = rd_mol.GetConformer().GetPositions()
    R, t = rigid_transform_Kabsch_3D(prior_coords.T, true_coords.T)
    rotated_rdkit_coords = ((R @ (prior_coords).T).T + t.squeeze())
    return rotated_rdkit_coords

def get_torsions_geo(mol_list):
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def mol2graph(mol, name = "test", radius=4, max_neighbors=None):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph

def coarsen_molecule(m):
    m = Chem.AddHs(m) #the GEOM dataset molecules have H's
    torsions = get_torsions_geo([m])
    if len(torsions) > 0:
        bond_break = [(b,c) for a,b,c,d in torsions]
        adj = Chem.rdmolops.GetAdjacencyMatrix(m)
        for r,c in bond_break:
            adj[r][c] = 0
            adj[c][r] = 0
        out = Chem.rdmolops.FragmentOnBonds(m,
                                        [m.GetBondBetweenAtoms(b[0], b[1]).GetIdx() for b in bond_break],
                                        addDummies=False) # determines fake bonds which adds fake atoms
        frags = Chem.GetMolFrags(out, asMols=True)
        frag_ids = Chem.GetMolFrags(out, asMols=False) #fragsMolAtomMapping = []
        frag_ids = [set(x) for x in frag_ids]
        cg_bonds = []
        cg_map = defaultdict(list)
        for start, end in bond_break:
            a = min(start, end)
            b = max(start, end)
            A, B = -1, -1
            for i, bead in enumerate(frag_ids):
                if a in bead:
                    A = i
                elif b in bead:
                    B = i
                if A > 0 and B > 0:
                    break
            cg_map[A].append(B)
            cg_map[B].append(A)
            cg_bonds.append((min(A,B), max(A,B)))
        return list(frags), frag_ids, adj, out, bond_break, cg_bonds, cg_map
    else:
        return [m], [0], Chem.rdmolops.GetAdjacencyMatrix(m), m, [], None, None

def create_pooling_graph(dgl_graph, frag_ids, latent_dim = 64, use_mean_node_features=True):
    N = len(frag_ids)
    n = dgl_graph.ndata['x'].shape[0]
    fine_coords = []
    coarse_coords = []
    # if use_mean_node_features:
    #     latent_dim += 5
#     M = np.zeros((N, n))
    chunks = []
    for bead, atom_ids in enumerate(frag_ids):
        subg = list(atom_ids)
        subg.sort()
        cc = dgl_graph.ndata['x'][subg,:].mean(dim=0).cpu().numpy().reshape(-1,3)
        fc = dgl_graph.ndata['x'][subg,:].cpu().numpy()
#         print(cc.shape, fc.shape)
        chunks.append(fc.shape[0])
        coarse_coords.append(cc)
        fine_coords.append(fc)
#         M[bead, list(atom_ids)] = 1
    fine_coords = np.concatenate(fine_coords, axis = 0)
    coarse_coords = np.concatenate(coarse_coords, axis = 0)
#     print(coarse_coords.shape, fine_coords.shape)
    coords = np.concatenate((fine_coords, coarse_coords), axis = 0)
    
    distance = spa.distance.cdist(coords, coords)
    src_list = []
    dst_list = []
    dist_list = []
    # mean_norm_list = [np.zeros((5,))]*n
    
    prev = 0
    for cg_bead in range(n, n+N):
        src = [prev + i for i in range(0, chunks[cg_bead-n])]
        dst = [cg_bead]*len(src)
        prev += len(src)
        src_list.extend(src)
        dst_list.extend(dst)
        
        valid_dist = list(distance[src, cg_bead])
        # print("\nd", cg_bead, src)
        dist_list.extend(valid_dist)
        # if len(src) <= 1:
        #      mean_norm_list.append([0,0,0,0,0])
        # else:
        #     valid_dist_np = distance[src, cg_bead]
        #     # print("V", valid_dist_np)
        #     sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        #     weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        #     # print("W", weights)
        #     assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        #     diff_vecs = coords[src, :] - coords[dst, :]  # (neigh_num, 3)
        #     mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        #     denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        #     # print("D",denominator)
        #     mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        #     # print("M", mean_vec_ratio_norm)
        #     mean_norm_list.append(mean_vec_ratio_norm)
        
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=n+N, idtype=torch.int32)

    graph.ndata['feat'] = torch.zeros((n+N, latent_dim))
    # graph.ndata['feat_pool'] = torch.zeros((n+N, latent_dim)) # for ECN updates
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(coords).astype(np.float32))
    # graph.ndata['x_pool'] = torch.from_numpy(np.array(coords).astype(np.float32))
    # graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph
        

def conditional_coarsen_3d(dgl_graph, frag_ids, cg_map, radius=4, max_neighbors=None, latent_dim_D = 64, latent_dim_F = 32,  use_mean_node_features = True):
    num_nodes = len(frag_ids)
    coords = []
    # if use_mean_node_features:
    #     latent_dim += 5
    M = np.zeros((num_nodes, dgl_graph.ndata['x'].shape[0]))
    for bead, atom_ids in enumerate(frag_ids):
        subg = list(atom_ids)
        subg.sort()
        coords.append(dgl_graph.ndata['x'][subg,:].mean(dim=0).cpu().numpy())
        M[bead, list(atom_ids)] = 1
        # TODO can scale by MW weighted_average = (A@W)/W.sum()
#         print(subg, coords[0].shape, coords)
    
    coords = np.asarray(coords)
    distance = spa.distance.cdist(coords, coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        assert dst != []
        
        required_dst = cg_map[i]
        for d in required_dst:
            if d not in dst:
                print("[Required] adding CG edge")
                dst.append(d)
        
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = coords[src, :] - coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = torch.zeros((num_nodes, latent_dim_D))
    graph.ndata['feat_pool'] = torch.zeros((num_nodes, latent_dim_D)) # for ECN updates
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(coords).astype(np.float32))
    graph.ndata['x_pool'] = torch.from_numpy(np.array(coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    graph.ndata['v'] = torch.zeros((num_nodes, latent_dim_F, 3))
    graph.ndata['M'] = torch.from_numpy(np.array(M).astype(np.float32))
    return graph

def get_coords(rd_mol):
    rd_conf = rd_mol.GetConformers()[0]
    positions = rd_conf.GetPositions()
    Z = []
    for position in positions:
        Z.append([*position])
    return np.asarray(Z)

def get_rdkit_coords(mol, seed = None):
    ps = AllChem.ETKDGv2()
    if seed is not None:
        ps.randomSeed = seed
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return np.asarray(lig_coords) #, dtype=torch.float32)

from rdkit.Chem.rdchem import BondType as BT
def get_bond(m, i, j):
    bond = m.GetBondBetweenAtoms(i,j)
    info = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    if bond == None:
        return -1e9 # not inf for stab
    else:
        typ = bond.GetBondType()
        return info[typ]
        
    
def mol2graphV2(mol, name = "test", radius=4, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    bond_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
        
        bonds = []
        for d in dst:
            bonds.append(get_bond(mol, int(i), int(d)))
        bond_list.extend(bonds)

    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.edata['bond_type'] = torch.from_numpy(np.asarray(bond_list).astype(np.float32))
    graph.edata['featV2'] = torch.cat((graph.edata['feat'],graph.edata['bond_type'].view(-1,1)), dim = -1)
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph

def mol2graphV3(mol, name = "test", radius=4, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    
    torsions = get_torsions_geo([mol])
    torsion_map = {}
    for b in torsions:
        torsion_map[(b[1], b[2])] = ([b[0], b[3]])
        
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    bond_list = []
    tort_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
        
        bonds = []
        torts = []
        for d in dst:
            bonds.append(get_bond(mol, int(i), int(d)))
            ii, dd = int(i), int(d)
            key = (min(ii, dd), max(ii, dd))
            if key in torsion_map:
                val = torsion_map[key]
                torts.append(GetDihedral(conf, [val[0], key[0], key[1], val[1]]))
            else:
                torts.append(0)
        bond_list.extend(bonds)
        tort_list.extend(torts)

    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.edata['bond_type'] = torch.from_numpy(np.asarray(bond_list).astype(np.float32))
    graph.edata['torsion_angle'] = torch.from_numpy(np.asarray(tort_list).astype(np.float32))
    graph.edata['featV2'] = torch.cat((graph.edata['feat'],graph.edata['bond_type'].view(-1,1)), dim = -1)
    graph.edata['featV3'] = torch.cat((graph.edata['feat'],graph.edata['bond_type'].view(-1,1), graph.edata['torsion_angle'].view(-1,1)), dim = -1)
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
#     if use_rdkit_coords:
#         graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph