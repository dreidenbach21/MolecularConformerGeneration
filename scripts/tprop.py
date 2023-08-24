import pickle

out_path = '/data/dreidenbach/data/torsional_diffusion/DRUGS/test_set/td_prop_pred_4_results_multi2.pkl'

with open(out_path, 'rb') as f:
    prop = pickle.load(f)

print(len(prop))
