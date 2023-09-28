import sys
# Print the current Python path
# print(sys.path)
import os
# print(os.environ['PYTHONPATH']) #sounce ./set_path.sh
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.data_utils_noar import * #generated_rdkit_confs
from model.vae_noar_angle_ot import VAENoArAngleOt
import datetime
from model.benchmarker import *
import glob
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
import networkx as nx
from utils.molecule_utils import *

# def collate(samples):
#     # import ipdb; ipdb.set_trace()
#     # A, B = map(list, zip(*samples))
#     A, B = zip(*[(t[0], t[1]) for sublist in samples for t in sublist])
# #     data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg
# #  .to('cuda:0') causes errors
#     nmols = [len(x) for x in samples]

#     A_graph = dgl.batch([x[0] for x in A])
#     geo_A = dgl.batch([x[1] for x in A])
#     angle_A = [x[2] for x in A]
#     angle_masks_A = [x[3] for x in A]
#     Ap = dgl.batch([x[4] for x in A])
#     A_cg = dgl.batch([x[5] for x in A])
#     geo_A_cg = dgl.batch([x[6] for x in A])
#     frag_ids = [x[7] for x in A]
    
#     B_graph = dgl.batch([x[0] for x in B])
#     geo_B = dgl.batch([x[1] for x in B])
#     angle_B = [x[2] for x in B]
#     angle_masks_B = [x[3] for x in B]
#     Bp = dgl.batch([x[4] for x in B])
#     B_cg = dgl.batch([x[5] for x in B])
#     geo_B_cg = dgl.batch([x[6] for x in B])
#     B_frag_ids = [x[7] for x in B]
#     return (A_graph, geo_A, angle_A, angle_masks_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, angle_B, angle_masks_B, Bp, B_cg, geo_B_cg, B_frag_ids), nmols
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

def dgl_to_mol(mol, data, mmff=False, rmsd=False, copy=True, key = 'x_cc'):
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)
    coords = data.ndata[key]
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except Exception as e:
            pass
    return mol

class PabloEscobar():
    def __init__(self, model):
        import datetime
        now = datetime.datetime.now()
        suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
        # weights = "DRUGS_noar_full_test_all_09-12_01-50-22_0" #"DRUGS_noar_full_test_one_fifth_save4_09-09_11-43-50_19" #
        # weights = "DRUGS_noar_ot_all_09-22_02-18-40_2" #"DRUGS_noar_one_fifth_ot4_09-21_16-58-11_1.pt"
        weights = "DRUGS_noar_ot_all_post_restart_09-26_23-36-23_2.pt"
        self.mole_name = "CC_DRUGS_testset_ot_" + weights  + suffix
        self.name = "CC_DRUGS_testset"
        # self.name = "CC_DRUGS_testset2"
        # self.name = "CC_DRUGS_testset_no_mmff"
        self.no_mmff = False #True
        self.model = model
        self.weights_path = "/home/dreidenbach/code/mcg/coagulation/scripts/model_ckpt/" + weights + ".pt"
        
        self.true_mols = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_mols.pkl'
        with open(self.true_mols, 'rb') as f:
            self.true_mols = pickle.load(f)
        self.valid_mols = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_smiles.csv'
        self.test_data = pd.read_csv(self.valid_mols)
        self.types = drugs_types
        self.save_dir = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set'
        self.batch_size = 400
        if False:
            chkpt = torch.load(self.weights_path)
            if "model" in chkpt:
                chkpt = chkpt['model']
            model.load_state_dict(chkpt, strict = False)
            print("CUDA CHECK", next(model.parameters()).is_cuda)
            print("# of Encoder Params = ", sum(p.numel()
                for p in model.encoder.parameters() if p.requires_grad))
            print("# of Decoder Params = ", sum(p.numel()
                for p in model.decoder.parameters() if p.requires_grad))
            print("# of VAE Params = ", sum(p.numel()
                for p in model.parameters() if p.requires_grad))
        
    def save(self):
        graphs, infos = [], []
        smiles = [x[0] for x in self.datapoints]
        data = [x[1] for x in self.datapoints]
        for A, B in data:
            data_A, geometry_graph_A, angle_graph_A, angle_masks_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids = A
            data_B, geometry_graph_B, angle_graph_B, angle_masks_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids = B
            infos.append((A_frag_ids, B_frag_ids, angle_graph_A, angle_graph_B, angle_masks_A, angle_masks_B))
            graphs.extend([data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg])
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_infos.bin', infos)
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_smiles.bin', smiles)
        # dgl.data.utils.save_info(self.save_dir + f'/{self.name}_problem_smiles.bin', self.problem_smiles)
        dgl.data.utils.save_graphs(self.save_dir + f'/{self.name}_graphs.bin', graphs)
        print("Saved Successfully", self.save_dir, self.name, len(self.datapoints))
        
    def load(self):
        print('Loading data ...')
        graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.name}_graphs.bin')
        info = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_infos.bin')
        smiles = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_smiles.bin')
        # self.problem_smiles = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_problem_smiles.bin')
        count = 0
        results_A, results_B = [], []
        for i in range(0, len(graphs), 10):
            AB = graphs[i: i+10]
            A_frag_ids, B_frag_ids, angle_graph_A, angle_graph_B, angle_masks_A, angle_masks_B = info[count]
            count += 1
            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
            results_A.append((data_A, geometry_graph_A, angle_graph_A, angle_masks_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
            results_B.append((data_B, geometry_graph_B, angle_graph_B, angle_masks_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
        data = [(a,b) for a,b in zip(results_A, results_B)]
        print("Loaded Successfully",  self.save_dir, self.name, len(data))

        # mdata = []
        # chunk = []
        # counts = []
        # size = -1
        # angles = None
        # count = 0
        # all_count = 0
        # for graphs in data:
        #     A, B = graphs
        #     if size == -1:
        #         size =  A[0].ndata['x'].shape[0]
        #         angles = A[2][0].keys()
        #     if size == A[0].ndata['x'].shape[0] and angles == A[2][0].keys():
        #         chunk.append(graphs)
        #     else:
        #         size =  A[0].ndata['x'].shape[0]
        #         angles = A[2][0].keys()
        #         if len(chunk) > 0:
        #             mdata.append(chunk)
        #         chunk = []
        # data = mdata
        return smiles, data
    
    def clean_confs(self, smi, confs):
        good_ids = []
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
        for i, c in enumerate(confs):
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
            if conf_smi == smi:
                good_ids.append(i)
        return [confs[i] for i in good_ids]
    
    def clean_true_mols(self):
        print("Cleaning True Molecules")
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            raw_smi = row['smiles']
            n_confs = row['n_conformers']
            smi = row['corrected_smiles']
            # if self.dataset == 'xl':
            #     raw_smi = smi
            self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            
    def has_cache(self):
        return os.path.exists(os.path.join(self.save_dir, f'{self.name}_graphs.bin'))
    
    def load_test_data(self):
        if self.has_cache():
            print("LOADING TEST SET FROM CACHE")
            self.clean_true_mols()
            self.smiles, data = self.load()
            print(f"{len(data)} Conformers Loaded")
        else:
            print("NO CACHE \n\n\n\n\n")
            self.build_test_data()
            self.save()
            # import ipdb; ipdb.set_trace()
            self.smiles = [x[0] for x in self.datapoints]
            data = [x[1] for x in self.datapoints]
        self.dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size= self.batch_size, shuffle=False, drop_last=False, num_workers=0 ,collate_fn = collate)
    
    def featurize_mol(self, smile, confs, factor = 1):
        name = smile

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        datas = []
        # nconfs = len(confs[:nlimit])
        for idx, conf in enumerate(confs):
            mol = Chem.AddHs(conf)
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
            print("SMILE does not match ground truth data")
            return None, None
        # import ipdb; ipdb.set_trace()
        mol = datas[0][-1]
        if self.no_mmff:
            rd_coords = generated_rdkit_confs_no_mmff(mol, factor*len(datas))
        else:
            rd_coords = generated_rdkit_confs(mol, factor*len(datas))
        if rd_coords == None:
            print("Cannot generate RDKit Coordinates")
            return None, None
        try:
            datas = datas+datas
            gt_datas = [(correct_mol[0], featurize_molecule(correct_mol[1], self.types, rdkit_coords = None)) for idx, correct_mol in enumerate(datas)]
            rd_datas = [(correct_mol[0], featurize_molecule(correct_mol[1], self.types, rdkit_coords = rd_coords[idx])) for idx, correct_mol in enumerate(datas)]
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
            test = 1
            gt_datas = None
            rd_datas = None
        return gt_datas, rd_datas
    
    def build_test_data(self):
        self.model_preds = defaultdict(list)
        self.problem_smiles = set()
        dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
        print("Buidling Test Set ...")
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            raw_smi = row['smiles']
            n_confs = row['n_conformers']
            smi = row['corrected_smiles']
            # import ipdb; ipdb.set_trace()
            print(smi)
            # if smi == "C#CCOCCOCCOCCNc1nc(N2CCN(C(=O)[C@H](CCC(=O)O)n3cc([C@H](N)CO)nn3)CC2)nc(N2CCN(C(=O)[C@H](CCC(=O)O)n3cc([C@@H]([NH3+])CO)nn3)CC2)n1":
            #     self.model_preds[smi] = [None]
            #     continue
            self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            if len(true_confs) == 0:
                print(f'poor ground truth conformers: {smi}')
                self.model_preds[smi] = [None]
                continue
            if '.' in smi:
                self.model_preds[smi] = [None]
                continue

            mol = Chem.MolFromSmiles(smi)
            if not mol:
                self.model_preds[smi] = [None]
                continue
            mol = true_confs[0]
            N = mol.GetNumAtoms()
            if not mol.HasSubstructMatch(dihedral_pattern) or N < 4:
                self.model_preds[smi] = [None]
                continue
            datas, data_Bs = self.featurize_mol(smi, true_confs, factor = 2)
            if not datas or len(datas) == 0 or not data_Bs or len(data_Bs) == 0:
                self.model_preds[smi] = [None]
                continue
            results_A = []
            results_B = []
            bad_idx_A, bad_idx_B = [], []
            angles = get_torsions_geo([mol])
            angle_masks_A = []
            for idx, (midx, data) in enumerate(datas):
                if not data:
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    continue
                mol = true_confs[midx] #mol_dic['conformers'][midx]['rd_mol']
                edge_mask, mask_rotate = get_transformation_mask2(mol, data, angles)
                mask_rotate_ten = torch.tensor(mask_rotate).T
                edge_mask_ten = torch.tensor(edge_mask)
                if np.sum(edge_mask) < 0.5: 
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    continue
                angle_masks_A.append(mask_rotate_ten)
                data.edata['mask_edges'] = edge_mask_ten
                try:
                    torsions_A, A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule_new(mol)
                    A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = 64, latent_dim_F = 32)
                except Exception as e:
                    print(e)
                    # self.failures['coarsening error'] += 1
                    # print(f"coarsening failure", mol_dic['smiles'])
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    continue
                geometry_graph_A = get_geometry_graph(mol)
                angle_graph_A = get_angle_graph(mol, torsions_A) 
                err = check_distances(data, geometry_graph_A)
                assert(err < 1e-3)
                Ap = create_pooling_graph(data, A_frag_ids)
                geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
                results_A.append((data, geometry_graph_A, angle_graph_A, angle_masks_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))

            angle_masks_B = []
            for idx, (midx, data_B) in enumerate(data_Bs):
                if idx in set(bad_idx_A):
                    bad_idx_B.append(idx)
                    results_B.append(None)
                    continue
                if not data_B:
                    # self.failures['featurize_mol_failed_B'] += 1
                    bad_idx_B.append(idx)
                    # print("BAD B", idx, len(data_Bs))
                    results_B.append(None)
                    continue
                    # return [None]
                mol = true_confs[midx] #mol_dic['conformers'][midx]['rd_mol']
                edge_mask, mask_rotate = get_transformation_mask2(mol, data_B, angles)
                mask_rotate_ten = torch.tensor(mask_rotate).T
                edge_mask_ten = torch.tensor(edge_mask)
                torsions_B, B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule_new(mol)
                B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = 64, latent_dim_F = 32)
                geometry_graph_B = get_geometry_graph(mol, data_B.ndata['x'].numpy())
                Bp = create_pooling_graph(data_B, B_frag_ids)
                geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
                err = check_distances(data_B, geometry_graph_B, True)
                angle_graph_B = get_angle_graph(mol, torsions_B)
                angle_masks_B.append(mask_rotate_ten)
                data_B.edata['mask_edges'] = edge_mask_ten
                assert(err < 1e-3)
                results_B.append((data_B, geometry_graph_B, angle_graph_B, angle_masks_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
            try:
                assert(len(results_A) == len(results_B)) # doubled in featurization
            except:
                print("mismatch A and B")
                import ipdb; ipdb.set_trace()
            bad_idx = set(bad_idx_A) | set(bad_idx_B)
            results_A = [x for idx, x in enumerate(results_A) if idx not in bad_idx]
            results_B = [x for idx, x in enumerate(results_B) if idx not in bad_idx]
            assert(len(results_A) == len(results_B))
            if len(results_A) == 0 or len(results_B) == 0:
                self.model_preds[smi] = [None]
                continue
            point_clouds_array = np.array([x[0].ndata['x_ref'].numpy() for x in results_B])
            unique_point_clouds_array = np.unique(point_clouds_array, axis=0)
            num_unique_point_clouds = unique_point_clouds_array.shape[0]
            # print("Unique RDKit", num_unique_point_clouds)
            # if num_unique_point_clouds != len(results_B):
            #     import ipdb; ipdb.set_trace()

            count = 0
            # first = results_B[:len(results_A)]
            # second = results_B[len(results_A):]
            # print(len(results_A), len(results_B))
            for a,b in zip(results_A, results_B):
                assert(a is not None and b is not None)
                self.model_preds[smi].append((a,b))
                # if count >= len(second):
                #     c = copy.deepcopy(first[0])
                # else:
                #     c = second[count]
                # self.model_preds[smi].append((copy.deepcopy(a), c))
                count += 1
                    
        self.datapoints = []
        for k, v in self.model_preds.items():
            if v[0] == None:
                continue
            self.datapoints.extend([(k, vv) for vv in v])
        print('Fetched', len(self.datapoints), 'mols successfully')
        
    def generate(self, rdkit_only = False):
        count = 0
        print(f"{len(self.dataloader)} batches")
        print("Testing")
        val_loss = 0
        molecules = []
        if rdkit_only:
            self.mole_name += "_rdkit_no_mmff"
            key = 'x'
        else:
            key = 'x_cc'
        with torch.no_grad():
            for A_batch, B_batch in self.dataloader:
            # for smi_idx, batch in tqdm(self.dataloader, total=len(self.dataloader)):
                # import ipdb; ipdb.set_trace()
                # A_batch,B_batch = batch
                print(f"Batch {count}")
                count+= 1
                A_graph, geo_A, angle_A, angle_mask_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch 
                B_graph, geo_B, angle_B, angle_mask_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
                if rdkit_only:
                    molecules.extend(dgl.unbatch(B_graph))
                else:
                    A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                    B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

                    generated_molecule, rdkit_reference, _, channel_selection_info, KL_terms, enc_out = self.model(B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=0, validation = True)
                    # generated_molecule = self.model.angle_forward(generated_molecule, angle_A, angle_mask_A, frag_ids)
                    molecules.extend(dgl.unbatch(generated_molecule.cpu()))
            
        self.final_confs = defaultdict(list)
        for smi, data in zip(self.smiles, molecules):
            self.final_confs[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True, key=key))
        with open(os.path.join(self.save_dir, f'{self.mole_name}_generated_molecules.pkl'), 'wb') as handle:
            pickle.dump(self.final_confs, handle)
        print("No Ar OT Generation Complete")


@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    model = VAENoArAngleOt(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, cfg.coordinates, device = "cuda").cuda()
    runner = PabloEscobar(model)
    runner.load_test_data()
    runner.generate(rdkit_only=False)
    print("Benchmark Test Complete")


if __name__ == "__main__":
    # mp.set_sharing_strategy('file_system')
    main()
