import sys
sys.path.insert(0, '/home/dreidenbach/code/mcg')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/model')
sys.path.insert(0, '/home/dreidenbach/code/mcg/coagulation/utils')
import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from tqdm import tqdm
import wandb, copy
# from utils.torsional_diffusion_data_all import featurize_mol, qm9_types, drugs_types, get_transformation_mask, check_distances
# from molecule_utils import *
# import dgl
from collections import defaultdict
from tqdm import tqdm

class RMSDRunner():
    def __init__(self, generated_mols = 'drugs_full_final_confs_gen3_rmsd_fix.pkl', true_mols = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_mols.pkl', valid_mols = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_smiles.csv', n_workers = 1, dataset = 'drugs',
                 D = 64, F = 32, save_dir = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set', batch_size = 2000, name = 'drugs_full'):
        with open(true_mols, 'rb') as f:
            self.true_mols = pickle.load(f)
        with open(save_dir + "/" + generated_mols, 'rb') as f:
            self.final_confs = pickle.load(f)
        self.threshold = np.arange(0, 2.5, .125)
        self.test_data = pd.read_csv(valid_mols)
        self.dataset = dataset
        self.n_workers = n_workers
        self.D, self.F = D, F
        if name is None:
            self.name = f'{dataset}_full'#_full_V3_check # dataste _80
        else:
            self.name = name
        self.batch_size = batch_size
        self.only_alignmol = False
        self.save_dir = save_dir
        # self.types = qm9_types if dataset == 'qm9' else drugs_types
        self.use_diffusion_angle_def = False
        self.clean_true_mols()
       
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

    def calculate(self):
        final_confs = self.final_confs
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
            
            true_confs = self.true_mols[corrected_smi]
            if len(true_confs) == 0:
                print(f'poor ground truth conformers: {corrected_smi}')
                continue
            
            if corrected_smi not in final_confs:
                if corrected_smi not in self.problem_smiles:
                    self.num_failures += 1
                    print('Cannot feed into Model', corrected_smi)
                else:
                    print('problematic ground truth not caught early on [CHECK]', corrected_smi)
                continue

            # true_mols[smi] = true_confs = self.clean_confs(corrected_smi, true_mols[smi])
            # true_confs = self.true_mols[smi]

            n_true = len(true_confs)
            n_model = len(final_confs[corrected_smi])
            results[(smi, corrected_smi)] = {
                'n_true': n_true,
                'n_model': n_model,
                'rmsd': np.nan * np.ones((n_true, n_model)),
                'coords': {}
            }
            # self.results = results
            for i_true in range(n_true):
                jobs.append((smi, corrected_smi, i_true))
                
        self.results = results
        random.shuffle(jobs)
        if self.n_workers > 1:
            p = Pool(self.n_workers)
            map_fn = p.imap_unordered
            p.__enter__()
        else:
            map_fn = map
        self.final_confs_temp = final_confs
        # conf_save_path = os.path.join(self.save_dir, f'{self.name}_final_confs_gen3_rmsd_fix_2.pkl')
        # with open(conf_save_path, 'wb') as handle:
        #     pickle.dump(final_confs, handle)
            
        for res in tqdm(map_fn(self.worker_fn, jobs), total=len(jobs)):
            self.populate_results(res)

        if self.n_workers > 1:
            p.__exit__(None, None, None)
        # ! previous code that worked below
        # self.final_confs_temp = final_confs
        # for job in tqdm(jobs, total=len(jobs)):
        #     self.populate_results(self.worker_fn(job))
        # import ipdb; ipdb.set_trace()
        use_wandb = False
        self.run(results, reduction='min', use_wandb = use_wandb)
        self.run(results, reduction='max', use_wandb = use_wandb)
        self.run(results, reduction='mean', use_wandb = use_wandb)
        self.run(results, reduction='std', use_wandb = use_wandb)
        import ipdb; ipdb.set_trace()
        # self.run(results, reduction='min', use_wandb = use_wandb)
        return results
        

    def calc_performance_stats(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.min(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.min(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision
    
    def calc_performance_stats_max(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.max(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.max(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.max(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.max(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision
    
    def calc_performance_stats_mean(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.mean(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.mean(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.mean(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.mean(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision
    
    def calc_performance_stats_std(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.std(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.std(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.std(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.std(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision

    def clean_confs(self, smi, confs):
        good_ids = []
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
        for i, c in enumerate(confs):
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
            if conf_smi == smi:
                good_ids.append(i)
        return [confs[i] for i in good_ids]

    def run(self, results, reduction = 'min', use_wandb = False, cut = 1):
        stats = []
        for res in results.values():
            res['rmsd'] = [x for idx, x in enumerate(res['rmsd']) if idx != 1] # normalize to TD
            if reduction == 'min':
                stats_ = self.calc_performance_stats(res['rmsd'][:, :max(1, res['rmsd'].shape[1]//cut)])
            elif reduction == 'max':
                stats_ = self.calc_performance_stats_max(res['rmsd'][:, :max(1, res['rmsd'].shape[1]//cut)])
            elif reduction == 'mean':
                stats_ = self.calc_performance_stats_mean(res['rmsd'][:, :max(1, res['rmsd'].shape[1]//cut)])
            elif reduction == 'std':
                stats_ = self.calc_performance_stats_std(res['rmsd'][:, :max(1, res['rmsd'].shape[1]//cut)])
            cr, mr, cp, mp = stats_
            stats.append(stats_)
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)
        for i, thresh in enumerate(self.threshold):
            coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * self.num_failures
            coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * self.num_failures
            if use_wandb:
                wandb.log({
                    f'{reduction} {thresh} Recall Coverage Mean': np.mean(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall Coverage Median': np.median(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall AMR Mean': np.nanmean(amr_recall),
                    f'{reduction} {thresh} Recall AMR Median': np.nanmedian(amr_recall),
                    f'{reduction} {thresh} Precision Coverage Mean': np.mean(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision Coverage Median': np.median(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision AMR Mean': np.nanmean(amr_precision),
                    f'{reduction} {thresh} Precision AMR Median': np.nanmedian(amr_precision),
                })
            else:
                print({
                    f'{reduction} {thresh} Recall Coverage Mean': np.mean(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall Coverage Median': np.median(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall AMR Mean': np.nanmean(amr_recall),
                    f'{reduction} {thresh} Recall AMR Median': np.nanmedian(amr_recall),
                    f'{reduction} {thresh} Precision Coverage Mean': np.mean(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision Coverage Median': np.median(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision AMR Mean': np.nanmean(amr_precision),
                    f'{reduction} {thresh} Precision AMR Median': np.nanmedian(amr_precision),
                })
        if use_wandb:
            wandb.log({
                f'{reduction} Conformer Sets Compared': len(results),
                f'{reduction} Model Failures': self.num_failures,
                f'{reduction} Additional Failures': np.isnan(amr_recall).sum()
            }) # can replace wandb log with
        else:
            print({
                f'{reduction} Conformer Sets Compared': len(results),
                f'{reduction} Model Failures': self.num_failures,
                f'{reduction} Additional Failures': np.isnan(amr_recall).sum()
            }) # can replace wandb log with
        return True
        
    def populate_results(self, res):
            smi, correct_smi, i_true, rmsds, coords = res
            self.results[(smi, correct_smi)]['rmsd'][i_true] = rmsds
            # import ipdb; ipdb.set_trace()
            self.results[(smi, correct_smi)]['coords'][i_true] = coords
            
    def worker_fn(self, job):
            smi, correct_smi, i_true = job
            # true_confs = self.true_mols[smi]
            true_confs = self.true_mols[correct_smi]
            model_confs = self.final_confs_temp[correct_smi]
            tc = true_confs[i_true]
            rmsds = []
            coords = []
            # import ipdb; ipdb.set_trace()
            for mc in model_confs:
                try:
                    if self.only_alignmol:
                        rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                    else:
                        # rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                        # tc_coords = torch.tensor(tc.GetConformer().GetPositions())
                        # mc_coords = torch.tensor(mc.GetConformer().GetPositions())
                        # tc_coords = self.align(tc_coords, mc_coords)
                        # rmsd = self.calculate_rmsd(tc_coords.numpy(), mc_coords.numpy())
                        a = tc.GetConformer().GetPositions()
                        b = mc.GetConformer().GetPositions()
                        # err = np.mean((a - b) ** 2)
                        # if err < 1e-7 :
                        #     print(f"[RMSD low crude error] {smi} {correct_smi} {i_true} = {err}")
                        #     import ipdb; ipdb.set_trace()
                        rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                        # if rmsd < 1e-2:
                        #     import ipdb; ipdb.set_trace()
                        # print("[Best RMSD , MSE ]", rmsd, err)
                    rmsds.append(rmsd)
                    coords.append((smi, a, b))
                except:
                    print('Additional failure', smi, correct_smi)
                    rmsds = [np.nan] * len(model_confs)
                    coords = [np.nan] * len(model_confs)
                    break
            return smi, correct_smi, i_true, rmsds, coords
    
    # def align(self, source, target):
    #     with torch.no_grad():
    #         lig_coords_pred = target
    #         lig_coords = source
    #         if source.shape[0] == 1:
    #             return source
    #         lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
    #         lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

    #         A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean) 
    #         A = A + torch.eye(A.shape[0]).to(A.device) * 1e-5 #added noise to help with gradients
    #         if torch.isnan(A).any() or torch.isinf(A).any():
    #             print("\n\n\n\n\n\n\n\n\n\nThe SVD tensor contains NaN or Inf values")
    #             return source
    #             # import ipdb; ipdb.set_trace()
    #         U, S, Vt = torch.linalg.svd(A)
    #         # corr_mat = torch.diag(1e-7 + torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
    #         corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
    #         rotation = (U @ corr_mat) @ Vt
    #         translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
    #     return (rotation @ lig_coords.t()).t() + translation
    
    def calculate_rmsd(self, array1, array2):
        # Calculate the squared differences
        squared_diff = np.square(array1 - array2)
        
        # Sum the squared differences along the axis=1
        sum_squared_diff = np.sum(squared_diff, axis=1)
        
        # Calculate the mean of the squared differences
        mean_squared_diff = np.mean(sum_squared_diff)
        
        # Calculate the square root of the mean squared differences
        rmsd = np.sqrt(mean_squared_diff)
        
        return rmsd


if __name__=='__main__':
    # import ipdb; ipdb.set_trace() 
    runner = RMSDRunner(n_workers = 10)
    runner.calculate()
    import ipdb; ipdb.set_trace()
    # with open('/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/drugs_full_cc_rmsd_results_coords.pkl', 'wb') as handle:
    #     pickle.dump(results, handle)
    
    
#     path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/'
# name = 'drugs_full_final_confs_gen3_rmsd_fix_rmsd_results.pkl'