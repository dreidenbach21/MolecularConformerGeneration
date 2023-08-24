import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="9"
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase
import time
import py3Dmol
import torch
print(torch.__version__, torch.cuda.is_available())
IPythonConsole.molSize = 400,400
IPythonConsole.drawOptions.addAtomIndices = True

sys.path.insert(0, '/home/dreidenbach/code/mcg')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/model')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/utils')
sys.path.insert(0, './TDC')


a = torch.randn((2,2)).cuda()

_3pbl = 'CCc1cc(c(c(c1O)C(=O)NC[C@@H]2CCC[N@]2CC)OC)Cl'
_2rgp = 'c1cc(cc(c1)F)Cn2c3ccc(cc3cn2)Nc4c(c(ncn4)N)CNN5CCCCC5'
_1iep = 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C'
_3eml = 'c1cc(oc1)c2nc3nc(nc(n3n2)N)NCCc4ccc(cc4)O'
_3ny8 = 'CC(C)CCC[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
_4rlu = 'c1cc(ccc1C=CC(=O)c2ccc(cc2O)O)O'
_4unn = 'COc1cccc(c1)C2=CC(=C(C(=O)N2)C#N)c3ccc(cc3)C(=O)O'
_5mo4 = 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)C(=O)Nc4cc(cc(c4)n5cc(nc5)C)C(F)(F)F'
_7l11 = 'CCCOc1cc(cc(c1)Cl)C2=CC(=CN(C2=O)c3cccnc3)c4ccccc4C#N'

from tdc import Oracle

ligand_name = {}
ligand_name[_3pbl] = '3pbl'
ligand_name[_2rgp] = '2rgp'
ligand_name[_1iep] = '1iep'
ligand_name[_3eml] = '3eml'
ligand_name[_3ny8] = '3ny8'
ligand_name[_4rlu] = '4rlu'
ligand_name[_4unn] = '4unn'
ligand_name[_5mo4] = '5mo4'
ligand_name[_7l11] = '7l11'

results = {}
for ligand in [_4rlu]: #[_3pbl, _2rgp, _1iep, _3eml, _3ny8, _4rlu, _4unn, _5mo4, _7l11]:
    oracle = Oracle(name = f'{ligand_name[ligand]}_docking_vina')
    for method in ['gm', 'cc', 'rd', 'td']:
        print(ligand_name[ligand], method)
        if method == 'cc':
            values = oracle(ligand, from_file = '/data/dreidenbach/data/torsional_diffusion/DRUGS/protein/protein_test_coarsen_conf_all.pkl')
        elif method == 'rd':
            values = oracle(ligand, from_file = '/data/dreidenbach/data/torsional_diffusion/DRUGS/protein/protein_test_rdkit_all.pkl')
        elif method == 'td':
            values = oracle(ligand, from_file = '/home/dreidenbach/code/mcg/torsional-diffusion/workdir/proteins/protein_20steps.pkl')
        elif method == 'gm':
            values = oracle(ligand, from_file = '/home/dreidenbach/code/mcg/GeoMol/protein_geomol.pkl')
        results[f'{ligand_name[ligand]}_{method}'] = values
        
with open("protein_results_4rlu.pkl", 'wb') as handle:
    pickle.dump(results, handle)
    
    # CUDA_VISIBLE_DEVICES=0 python generate_confs.py --trained_model_dir trained_models/drugs/ --test_csv /home/dreidenbach/code/notebooks/3pbl.csv --dataset drugs --out protein_geomol
    