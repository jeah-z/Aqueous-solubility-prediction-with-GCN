import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem


validation_split = .1
shuffle_dataset = True
random_seed = 42
parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", help="dataset to train")
parser.add_argument("--eval_dataset", help="dataset eval")
args = parser.parse_args()
train_dataset = args.train_dataset
eval_dataset = args.eval_dataset


delaney = pd.read_csv(eval_dataset + ".csv", skiprows=1,
                      names=['id', 'measured', 'predicted', 'SMILES'])
dataset_size = len(delaney)
invalid_id = []
for i in range(dataset_size):
    smi = delaney.loc[i]['SMILES']
    try:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)

    except:
        print(smi + "was not valid SMILES\n")
        invalid_id.append(i)
delaney.drop(labels=invalid_id, axis=0)

def open_file(path, skipfirst=True, split=False):
    with  open(path) as smi: 
        if skipfirst:           # gzip.open(path) as smi:
            smi.readline()
        lines = smi.readlines()
        # for i in range(len(lines)):
        if split==True:
            for i in range(len(lines)):
                lines[i] = lines[i].split()
                lines[i][1] = float(lines[i][1])
            print(str(lines) + "\n")
    return lines
mean_std=open_file(train_dataset+'_mean_std.txt',False,True)
train_mean = mean_std[0][1]
train_std = mean_std[1][1]
delaney['measured'] = (
    delaney['measured'] - train_mean) / train_std


delaney.to_csv(train_dataset+'_'+eval_dataset+"_cross.csv", index=False)
