import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from tqdm import tqdm
import wandb, copy
from utils.torsional_diffusion_data_all import featurize_mol, qm9_types, drugs_types, get_transformation_mask, check_distances
from molecule_utils import *
import dgl
from collections import defaultdict
from tqdm import tqdm

# parser = ArgumentParser()
# parser.add_argument('--confs', type=str, required=True, help='Path to pickle file with generated conformers')
# parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles_corrected.csv', help='Path to csv file with list of smiles')
# parser.add_argument('--true_mols', type=str, default='./data/DRUGS/test_mols.pkl', help='Path to pickle file with ground truth conformers')
# parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
# parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
# parser.add_argument('--dataset', type=str, default="drugs", help='Dataset: drugs, qm9 and xl')
# parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
# parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
# args = parser.parse_args()

"""
    Evaluates the RMSD of some generated conformers w.r.t. the given set of ground truth
    Part of the code taken from GeoMol https://github.com/PattanaikL/GeoMol
"""
# with open(args.confs, 'rb') as f:
#     model_preds = pickle.load(f)

# test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
# with open(args.true_mols, 'rb') as f:
#     true_mols = pickle.load(f)
# threshold = threshold_ranges = np.arange(0, 2.5, .125)
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
    # try:
    #     if rmsd:
    #         mol.rmsd = AllChem.GetBestRMS(
    #             Chem.RemoveHs(data.seed_mol),
    #             Chem.RemoveHs(mol)
    #         )
    #     # mol.total_perturb = data.total_perturb
    # except:
    #     pass
    # mol.n_rotable_bonds = data.edge_mask.sum()
    return mol
    # if not copy: return mol
    # import ipdb; ipdb.set_trace()
    # return copy.deepcopy(mol)

def collate(samples):
    A, B = map(list, zip(*samples))
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    Ap = dgl.batch([x[2] for x in A])
    A_cg = dgl.batch([x[3] for x in A])
    geo_A_cg = dgl.batch([x[4] for x in A])
    frag_ids = [x[5] for x in A]
    #
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    Bp = dgl.batch([x[2] for x in B])
    B_cg = dgl.batch([x[3] for x in B])
    geo_B_cg = dgl.batch([x[4] for x in B])
    B_frag_ids = [x[5] for x in B]
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids)

class BenchmarkRunner():
    def __init__(self, true_mols = '/home/dreidenbach/data/torsional_diffusion/QM9/test_mols.pkl', valid_mols = '/home/dreidenbach/data/torsional_diffusion/QM9/test_smiles.csv', n_workers = 1, dataset = 'qm9',
                 D = 64, F = 32, save_dir = '/home/dreidenbach/data/torsional_diffusion/QM9', batch_size = 2000):
        with open(true_mols, 'rb') as f:
            self.true_mols = pickle.load(f)
        self.threshold = np.arange(0, 2.5, .125)
        self.test_data = pd.read_csv(valid_mols)
        self.dataset = dataset
        self.n_workers = n_workers
        self.D, self.F = D, F
        self.name = f'{dataset}_test_set_benchmark_fixed'
        self.batch_size = batch_size
        self.only_alignmol = False
        self.save_dir = save_dir
        self.types = qm9_types if dataset == 'qm9' else drugs_types
        self.use_diffusion_angle_def = False
        if self.has_cache():
            self.clean_true_mols()
            self.smiles, data = self.load()
        else:
            self.build_test_dataset()
            self.save() 
            self.smiles = [x[0] for x in self.datapoints]
            data = [x[1] for x in self.datapoints]
        self.dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size= self.batch_size, shuffle=False, drop_last=False, num_workers=1 ,collate_fn = collate)
    
    def clean_true_mols(self):
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            # if smi_idx < 950:
                # continue
            raw_smi = row['smiles']
            n_confs = row['n_conformers']
            smi = row['corrected_smiles']
            if self.dataset == 'xl':
                raw_smi = smi
            # self.true_mols[raw_smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            
    def build_test_dataset(self, confs_per_mol = None):
        self.model_preds = defaultdict(list)
        self.problem_smiles = set()
        # import ipdb; ipdb.set_trace()
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            # if smi_idx < 950:
                # continue
            raw_smi = row['smiles']
            n_confs = row['n_conformers']
            smi = row['corrected_smiles']
            if self.dataset == 'xl':
                raw_smi = smi
            # self.true_mols[raw_smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            # import ipdb; ipdb.set_trace()
            if len(true_confs) == 0:
                print(f'poor ground truth conformers: {smi}')
                self.model_preds[smi] = [None]
                self.problem_smiles.add(smi)
                continue
            # if confs_per_mol:
            #     duplicate = confs_per_mol
            # else:
            #     duplicate = 2*len(true_confs) #n_confs
            # moles = [self.true_mols[raw_smi]] * duplicate
            # moles = []
            # for mol in true_confs:
            #     moles.extend([mol, mol])
            # datas = self.featurize_mol(smi, moles)
            datas = self.featurize_mol(smi, true_confs)
            # datas = []
            # for d in datas_:
            #     datas.extend([d, copy.deepcopy(d)])
            # import ipdb; ipdb.set_trace()
            bad_idx_A = []
            results_A = []
            for idx, data in enumerate(datas):
                mol = true_confs[idx]
                edge_mask, mask_rotate = get_transformation_mask(mol, data)
                if np.sum(edge_mask) < 0.5:
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    self.problem_smiles.add(smi)
                    continue
                A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
                try:
                    A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
                except:
                    print("AddHs Data Issue: ground truth molecule does not have full Hs", idx, smi)
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    self.problem_smiles.add(smi)
                    continue
                    # import ipdb; ipdb.set_trace()
                    # A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
                    # A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
                geometry_graph_A = get_geometry_graph(mol)
                err = check_distances(data, geometry_graph_A)
                if err.item() > 1e-3:
                    import ipdb; ipdb.set_trace()
                    data = self.featurize_mol(mol_dic)
                Ap = create_pooling_graph(data, A_frag_ids)
                geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
                results_A.append((data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
            # import ipdb; ipdb.set_trace()
            if len(results_A) == len(bad_idx_A):
                self.model_preds[smi] = [None]
                self.problem_smiles.add(smi)
                continue
            data_Bs = self.featurize_mol(smi, true_confs, use_rdkit_coords = True)
            # data_Bs2 = self.featurize_mol(smi, copy.deepcopy(true_confs), use_rdkit_coords = True)
            # for aa ,bb in zip(data_Bs, data_Bs2):
            #     if aa is not None and bb is not None:
            #         import ipdb; ipdb.set_trace()
            #         c = 1
            # import ipdb; ipdb.set_trace()
            bad_idx_B = []
            results_B = []
            # data_Bs = []
            # for d in datas_:
            #     data_Bs.extend([d, copy.deepcopy(d)])
            for idx, data_B in enumerate(data_Bs):
                if idx in set(bad_idx_A):
                    bad_idx_B.append(idx)
                    results_B.append(None)
                    continue
                if not data_B:
                    print('Cannot RDKit Featurize', idx, smi)
                    bad_idx_B.append(idx)
                    results_B.append(None)
                    # self.problem_smiles.add(smi)
                    continue
                    # return False
                mol = true_confs[idx]
                B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
                B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
    #             geometry_graph_B = copy.deepcopy(geometry_graph_A) #get_geometry_graph(mol)
                geometry_graph_B = get_geometry_graph(mol)
                Bp = create_pooling_graph(data_B, B_frag_ids)
                geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
                err = check_distances(data_B, geometry_graph_B, True)
                if err.item() > 1e-3:
                    import ipdb; ipdb.set_trace()
                    data_B = self.featurize_mol(mol_dic, use_rdkit_coords = True)
                results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
            try:
                assert(len(results_A) == len(results_B))
            except:
                import ipdb; ipdb.set_trace()
            bad_idx = set(bad_idx_A) | set(bad_idx_B)
            # import ipdb; ipdb.set_trace()
            results_A = [x for idx, x in enumerate(results_A) if idx not in bad_idx]
            results_B = [x for idx, x in enumerate(results_B) if idx not in bad_idx]
            assert(len(results_A) == len(results_B))
            if len(results_A) == 0 or len(results_B) == 0:
                self.model_preds[smi] = [None]
                # self.problem_smiles.add(smi)
            # self.model_preds[smi] = [(a,b) for a,b in zip(results_A, results_B)]
            for a,b in zip(results_A, results_B):
                self.model_preds[smi].append((a,b))
                self.model_preds[smi].append((copy.deepcopy(a),copy.deepcopy(b)))
        self.datapoints = []
        for k, v in self.model_preds.items():
            if v[0] == None:
                continue
            self.datapoints.extend([(k, vv) for vv in v])
        print('Fetched', len(self.datapoints), 'mols successfully')
        print('Example', self.datapoints[0])
        
    def featurize_mol(self, smi, moles, use_rdkit_coords = False):
        name = smi
        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)
        datas = []
        early_kill = False
        for mol in moles:
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue
            if conf_canonical_smi != canonical_smi or early_kill:
                datas.append(None)
                continue
            # pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            correct_mol = mol
            check_ = correct_mol.GetConformer().GetPositions()
            mol_features = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords)
            if mol_features is not None:
                try:
                    cmpt = correct_mol.GetConformer().GetPositions()
                    if use_rdkit_coords:
                        # a = cmpt-np.mean(cmpt, axis = 0)
                        # b = mol_features.ndata['x_ref'].numpy()
                        # assert(np.mean((a - b) ** 2) < 1e-7 ) # This fails since the featurization aligns the rdkit so the MSE is not preserved
                        cmpt = check_
                    a = cmpt-np.mean(cmpt, axis = 0)
                    b = mol_features.ndata['x_true'].numpy()
                    assert(np.mean((a - b) ** 2) < 1e-7 )
                except:
                    import ipdb; ipdb.set_trace()
                    mol_features = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords)
                
            datas.append(mol_features)
            if mol_features is None:
                print(f"Skipping {len(moles)-1} since I am {use_rdkit_coords} using RDKit and I am getting a Featurization error")
                early_kill = True
        # if use_rdkit_coords:
        #     import ipdb; ipdb.set_trace()
        return datas
            
    def generate(self, model, rdkit_only = False, save = True):
        if not rdkit_only:
            if save and os.path.exists(os.path.join(self.save_dir, f'{self.name}_random_weights_gen.bin')):
                molecules, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.name}_random_weights_gen.bin')
            else:
                molecules = []
                with torch.no_grad():
                    for A_batch, B_batch in self.dataloader:
                        A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                        B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids= B_batch

                        A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                        B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')
                        # import ipdb; ipdb.set_trace()
                        generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = model(
                                B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=0, validation = True)
                        # ipdb.set_trace()
                        # loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, AR_loss, step=0, log_latent_stats = False)
                        # train_loss_log.append(losses)
                        # losses['Test Loss'] = loss.cpu()
                        # wandb.log({'test_' + key: value for key, value in losses.items()})
                        molecules.extend(dgl.unbatch(generated_molecule.cpu()))
                if save:
                    dgl.data.utils.save_graphs(self.save_dir + f'/{self.name}_random_weights_gen.bin', molecules)
            self.final_confs = defaultdict(list)
            # self.final_confs_rdkit = defaultdict(list)
            for smi, data in zip(self.smiles, molecules):
                self.final_confs[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True))
                # self.final_confs_rdkit[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True, key = 'x_ref'))
            self.results_model = self.calculate(self.final_confs)
            # self.results_rdkit = self.calculate(self.final_confs_rdkit)
        else:
            molecules = []
            for A_batch, B_batch in self.dataloader:
                A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
                molecules.extend(dgl.unbatch(B_graph))
            self.final_confs_rdkit = defaultdict(list)
            for smi, data in zip(self.smiles, molecules):
                self.final_confs_rdkit[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True, key = 'x_ref'))
            self.results_rdkit = self.calculate(self.final_confs_rdkit)
                
            
        
        
    def calculate(self, final_confs):
        rdkit_smiles = self.test_data.smiles.values
        corrected_smiles = self.test_data.corrected_smiles.values
        self.num_failures = 0
        results = {}
        jobs = []
        # for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            # if smi_idx < 950:
                # continue
            smi = row['smiles']
            n_confs = row['n_conformers']
            corrected_smi = row['corrected_smiles']
            if self.dataset == 'xl':
                smi = corrected_smi
            if corrected_smi not in final_confs:
                if corrected_smi not in self.problem_smiles:
                    self.num_failures += 1
                    print('model failure error in RDKit or elsewhere', corrected_smi)
                else:
                    print('problematic ground truth', corrected_smi)
                continue

            # true_mols[smi] = true_confs = self.clean_confs(corrected_smi, true_mols[smi])
            # true_confs = self.true_mols[smi]
            true_confs = self.true_mols[corrected_smi]

            if len(true_confs) == 0:
                print(f'poor ground truth conformers: {corrected_smi}')
                continue

            n_true = len(true_confs)
            n_model = len(final_confs[corrected_smi])
            results[(smi, corrected_smi)] = {
                'n_true': n_true,
                'n_model': n_model,
                'rmsd': np.nan * np.ones((n_true, n_model))
            }
            self.results = results
            for i_true in range(n_true):
                jobs.append((smi, corrected_smi, i_true))

        random.shuffle(jobs)
        if self.n_workers > 1:
            p = Pool(self.n_workers)
            map_fn = p.imap_unordered
            p.__enter__()
        else:
            map_fn = map
        self.final_confs_temp = final_confs
        for res in tqdm(map_fn(self.worker_fn, jobs), total=len(jobs)):
            self.populate_results(res)

        if self.n_workers > 1:
            p.__exit__(None, None, None)
        self.run(results)
        # import ipdb; ipdb.set_trace()
        return results
        

    def calc_performance_stats(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.min(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.min(axis=0).mean()

        return coverage_recall, amr_recall, coverage_precision, amr_precision

    def clean_confs(self, smi, confs):
        good_ids = []
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
        for i, c in enumerate(confs):
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
            if conf_smi == smi:
                good_ids.append(i)
        return [confs[i] for i in good_ids]

    def run(self, results):
        stats = []
        for res in results.values():
            stats_ = self.calc_performance_stats(res['rmsd'])
            cr, mr, cp, mp = stats_
            stats.append(stats_)
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)
        for i, thresh in enumerate(self.threshold):
            coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * self.num_failures
            coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * self.num_failures
            # wandb.log({
            #     f'{thresh} Recall Coverage Mean': np.mean(coverage_recall_vals) * 100,
            #     f'{thresh} Recall Coverage Median': np.median(coverage_recall_vals) * 100,
            #     f'{thresh} Recall AMR Mean': np.nanmean(amr_recall),
            #     f'{thresh} Recall AMR Median': np.nanmedian(amr_recall),
            #     f'{thresh} Precision Coverage Mean': np.mean(coverage_precision_vals) * 100,
            #     f'{thresh} Precision Coverage Median': np.median(coverage_precision_vals) * 100,
            #     f'{thresh} Precision AMR Mean': np.nanmean(amr_precision),
            #     f'{thresh} Precision AMR Median': np.nanmedian(amr_precision),
            # })
            print({
                f'{thresh} Recall Coverage Mean': np.mean(coverage_recall_vals) * 100,
                f'{thresh} Recall Coverage Median': np.median(coverage_recall_vals) * 100,
                f'{thresh} Recall AMR Mean': np.nanmean(amr_recall),
                f'{thresh} Recall AMR Median': np.nanmedian(amr_recall),
                f'{thresh} Precision Coverage Mean': np.mean(coverage_precision_vals) * 100,
                f'{thresh} Precision Coverage Median': np.median(coverage_precision_vals) * 100,
                f'{thresh} Precision AMR Mean': np.nanmean(amr_precision),
                f'{thresh} Precision AMR Median': np.nanmedian(amr_precision),
            })
            print("\n\n")
            # print('threshold', thresh)
            # coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * num_failures
            # coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * num_failures
            # print(f'Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
            # print(f'Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
            # print(f'Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
            # print(f'Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')
        # import ipdb; ipdb.set_trace()
        # wandb.log({
        #     'Conformer Sets Compared': len(self.results),
        #     'Model Failures': self.num_failures,
        #     'Addittional Failures': np.isnan(amr_recall).sum()
        # })
        print({
            'Conformer Sets Compared': len(results),
            'Model Failures': self.num_failures,
            'Addittional Failures': np.isnan(amr_recall).sum()
        })
        print("\n\n")
        # print(len(results), 'conformer sets compared', num_failures, 'model failures', np.isnan(amr_recall).sum(),
            # 'additional failures')
        return True
    
    def save(self):
        graphs, infos = [], []
        smiles = [x[0] for x in self.datapoints]
        data = [x[1] for x in self.datapoints]
        for A, B in data:
            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids = A
            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids = B
            infos.append((A_frag_ids, B_frag_ids))
            graphs.extend([data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg])
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_infos.bin', infos)
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_smiles.bin', smiles)
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_problem_smiles.bin', self.problem_smiles)
        dgl.data.utils.save_graphs(self.save_dir + f'/{self.name}_graphs.bin', graphs)
        print("Saved Successfully", self.save_dir, self.name, len(self.datapoints))
    
    def load(self):
        graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.name}_graphs.bin')
        info = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_infos.bin')
        smiles = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_smiles.bin')
        self.problem_smiles = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_problem_smiles.bin')
        count = 0
        results_A, results_B = [], []
        for i in range(0, len(graphs), 10):
            AB = graphs[i: i+10]
            A_frag_ids, B_frag_ids = info[count]
            count += 1
            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
            results_A.append((data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
            results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
        data = [(a,b) for a,b in zip(results_A, results_B)]
        print("Loaded Successfully",  self.save_dir, self.name, len(data))
        return smiles, data
    
    def has_cache(self):
         return os.path.exists(os.path.join(self.save_dir, f'{self.name}_graphs.bin'))
        
    def populate_results(self, res):
            smi, correct_smi, i_true, rmsds = res
            self.results[(smi, correct_smi)]['rmsd'][i_true] = rmsds
            
    def worker_fn(self, job):
            smi, correct_smi, i_true = job
            # true_confs = self.true_mols[smi]
            true_confs = self.true_mols[correct_smi]
            model_confs = self.final_confs_temp[correct_smi]
            tc = true_confs[i_true]
            rmsds = []
            for mc in model_confs:
                try:
                    if self.only_alignmol:
                        rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                    else:
                        a = tc.GetConformer().GetPositions()
                        b = mc.GetConformer().GetPositions()
                        err = np.mean((a - b) ** 2)
                        if err < 1e-7 :
                            print(f"[RMSD low crude error] {smi} {correct_smi} {i_true} = {err}")
                        rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                    rmsds.append(rmsd)
                except:
                    print('Additional failure', smi, correct_smi)
                    rmsds = [np.nan] * len(model_confs)
                    break
            return smi, correct_smi, i_true, rmsds