import os
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
# os.environ["OMP_STACKSIZE"]="4G"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="9"
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tqdm, multiprocessing
from collections import defaultdict, Counter
import subprocess

import rdkit
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, rdMolTransforms, rdmolfiles
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
import time
# import py3Dmol
import torch
print(torch.__version__, torch.cuda.is_available())
# IPythonConsole.molSize = 400,400
# IPythonConsole.drawOptions.addAtomIndices = True

sys.path.insert(0, '/home/dreidenbach/code/mcg')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/model')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/utils')
sys.path.insert(0, './TDC')


a = torch.randn((2,2)).cuda()

# out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/cc_prop_pred_4_results.pkl'
# out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/gt_prop_pred_4_results.pkl'
# out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/td_prop_pred_4_results_multi3.pkl' #!first IP 1.32
# out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/gm_prop_pred_results_multi.pkl'
# out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/gd_prop_pred_results_multi.pkl'
out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/rd_prop_pred_results_multi.pkl'

# path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/'
# cc = 'cc_2conf_property_prediction.pkl'
# gt = 'cc_2conf_property_prediction_ground_truth.pkl'
# td = 'td_property_prediction.pkl'
# with open(path + td, 'rb') as f:
#     prop = pickle.load(f)
    
    
# mol_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_mols.pkl' #TRUE
# mol_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/drugs_full_final_confs_gen3_rmsd_fix.pkl' #CC
# mol_path = '/home/dreidenbach/code/mcg/torsional-diffusion/workdir/drugs_default/drugs_steps20.pkl' #TD
# mol_path = '/home/dreidenbach/code/mcg/torsional-diffusion/workdir/geodiff_drugs/samples_all_labelled.pkl' #geodiff_path
mol_path = '/home/dreidenbach/code/mcg/GeoMol/trained_models/drugs/test_mols.pkl' #geomol_path
# with open('/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/property_smiles4.pkl', 'rb') as f:
#     smiles = pickle.load(f)

with open('/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/E_smiles_comp.pkl', 'rb') as f:
     smiles = pickle.load(f)
     
print('Optimizing', len(smiles), 'mols')
with open(mol_path, 'rb') as f:
    all_mols = pickle.load(f)
    
# smiles = [x[0] for x in smiles]
# smiles = [x[1] for x in smiles] #[x[0] for x in smiles]
prop = {key: value for key, value in all_mols.items() if key in smiles}
# # all_true_mols = pickle.load(open(true_mols, 'rb'))
# # true_mols = {key: value for key, value in all_true_mols.items() if key in smiles1}
print("loaded", len(prop), "molecules")
    
from ase import Atoms
from xtb.ase.calculator import XTB
from ase.optimize import BFGS

# my_dir = f"/tmp/{os.getpid()}"
# if not os.path.isdir(my_dir):
#     os.mkdir(my_dir)


def xtb_energy(mol, path_xtb, water=False, dipole=False):
    my_dir = f"/tmp/{os.getpid()}"
    if not os.path.isdir(my_dir):
        os.mkdir(my_dir)
    path = f"/tmp/{os.getpid()}.xyz"
    rdmolfiles.MolToXYZFile(mol, path)
    cmd = [path_xtb, path, '--iterations', str(1000)]
    if water:
        cmd += ['--alpb', 'water']
    if dipole:
        cmd += ['--dipole']
    n_tries = 3
    result = {}
    for i in range(n_tries):
        try:
            out = subprocess.check_output(cmd, stderr=open('/dev/null', 'w'), cwd=my_dir)
            break
        except subprocess.CalledProcessError as e:
            if i == n_tries-1:
                print('xtb_energy did not converge')
                return result #print(e.returncode, e.output)
    if dipole:
        dipole = [line for line in out.split(b'\n') if b'full' in line][1]
        result['dipole'] = float(dipole.split()[-1])
        
    runtime = out.split(b'\n')[-8].split()
    result['runtime'] = float(runtime[-2]) + 60*float(runtime[-4]) + 3600*float(runtime[-6]) + 86400*float(runtime[-8])                    
    
    energy = [line for line in out.split(b'\n') if b'TOTAL ENERGY' in line]
    result['energy'] = 627.509 * float(energy[0].split()[3])
    
    gap = [line for line in out.split(b'\n') if b'HOMO-LUMO GAP' in line]
    result['gap'] = 23.06 * float(gap[0].split()[3])
    
    return result

def relax(mol): #TODO run this in parralel over all confs for each mol.key
    try:
        # calc = XTB(method = 'GFN2-xTB')
        use_rdkit = False
        if use_rdkit:
            mol = Chem.MolFromSmiles(mol)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        atoms = Atoms(numbers = [x.GetAtomicNum() for x in mol.GetAtoms()], positions = mol.GetConformer().GetPositions())
        atoms.calc = calc
        opt = BFGS(atoms)
        try:
            opt.run(fmax = 0.03)
        except:
            return None
        coords = atoms.positions
        for i in range(coords.shape[0]):
            mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        return mol
    except:
        print("failed relax")
        return None

# mol = prop['C[C@H](CCc1ccccc1)NC[C@H](O)c1ccc(O)c(C(N)=O)c1'][0]

relaxed_mols = {}
for smi in prop.keys():
    print("Relaxing", smi)
    
    mols = prop[smi][:32]
    # mols = [smi for _ in range(len(mols))] #! for RDKit
    parallel = True
    calc = XTB(method = 'GFN2-xTB')
    if parallel:
        num_threads = 32
        with multiprocessing.Pool(processes=num_threads) as pool:
            relaxed = pool.map(relax, mols)
    else:
        relaxed = []
        for mol in mols:
            print("Relaxing", smi)
            relaxed.append(relax(mol))
    relaxed_mols[smi] = [x for x in relaxed if x != None]
    print(len(relaxed_mols[smi]), "out of ", len(mols))

new_mols = {}
for smi in relaxed_mols.keys():
    print("Predicting", smi)
    confs = relaxed_mols[smi]
    new_confs = []
    for conf in tqdm.tqdm(confs):
        res = xtb_energy(conf, dipole=True, path_xtb='/home/dreidenbach/anaconda3/envs/mcg2/bin/xtb')
        if not res: continue
        conf.xtb_energy, conf.xtb_dipole, conf.xtb_gap, conf.xtb_runtime = res['energy'], res['dipole'], res['gap'], res['runtime']
        new_confs.append(conf)
    new_mols[smi] = new_confs
    
open(out_path, 'wb').write(pickle.dumps(new_mols))
print("Done")
# # ! TD force batch
# new_mols = {}
# # keys = list(prop.keys())
# batch_size = 1
# calc = XTB(method = 'GFN2-xTB')
# for idx in range(66+70, len(prop)//batch_size):
#     print("batch", idx) # 16 32 48 64 79 95 smiles 69 smiles2
#     # if idx == 16:
#     #     continue
#     relaxed_mols = {}
#     batch = smiles[idx: (idx+1)*batch_size]
#     for smi in batch:
#         print("Relaxing", smi)
#         mols = prop[smi][:32]
#         # parallel = True
#         num_threads = 32
#         with multiprocessing.Pool(processes=num_threads) as pool:
#             relaxed = pool.map(relax, mols)
#         relaxed_mols[smi] = [x for x in relaxed if x != None]
#         print(len(relaxed_mols[smi]), "out of ", len(mols))


#     for smi in relaxed_mols.keys():
#         print("Predicting", smi)
#         confs = relaxed_mols[smi]
#         new_confs = []
#         for conf in tqdm.tqdm(confs):
#             res = xtb_energy(conf, dipole=True, path_xtb='/home/dreidenbach/anaconda3/envs/mcg2/bin/xtb')
#             if not res: continue
#             conf.xtb_energy, conf.xtb_dipole, conf.xtb_gap, conf.xtb_runtime = res['energy'], res['dipole'], res['gap'], res['runtime']
#             new_confs.append(conf)
#         new_mols[smi] = new_confs
    
#     open(out_path, 'wb').write(pickle.dumps(new_mols))
#     print("save", idx)
# print("Done")