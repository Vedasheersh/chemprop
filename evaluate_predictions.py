import sys
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def _calc_metrics(target, pred):
    return {'r2': r2_score(target, pred),
           'mae': mean_absolute_error(target,pred),
           'mse': mean_squared_error(target,pred)}

PARAMETER = 'ki'
PREDS_DIR = f'/home/ubuntu/mychemprop/experiments/{PARAMETER}/seq_random_evi_ens5_test/'
DATA_DIR = '/home/ubuntu/mychemprop/CatPred-DB/data/processed/splits_wpdbs/'

PREDFILE_PREFIX = 'test_preds_unc_evi_mvewt_' #seq80.csv
DATAFILE_PREFIX = f'{PARAMETER}-random_test_sequence_' #80cluster.csv

TARGETCOL = f'log10{PARAMETER}_max' if PARAMETER=='kcat' else f'log10{PARAMETER}_mean'
STDEVCOL = f'{TARGETCOL}_evidential_total_mve_weighting_stdev'

SMILESCOL = 'reactant_smiles' if PARAMETER=='kcat' else 'substrate_smiles'

RANGE = [20,40,60,80,99]

for R in RANGE:
    datafile = f'{DATA_DIR}/{DATAFILE_PREFIX}{R}cluster.csv'
    predsfile = f'{PREDS_DIR}/{PREDFILE_PREFIX}seq{R}.csv'
    data_df = pd.read_csv(datafile)
    data_df.index = data_df[SMILESCOL] + data_df['sequence']
    preds_df = pd.read_csv(predsfile)
    preds_df.index = preds_df[SMILESCOL] + preds_df['sequence']
    pred = preds_df[TARGETCOL]
    target = [data_df.loc[ind][TARGETCOL] for ind in preds_df.index]
    std = preds_df[STDEVCOL]
    print(R, _calc_metrics(target,pred))
    # break