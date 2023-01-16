from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import os
import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import pickle




def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0


opt = {"targets": "data/ChEMBL/tgt-train",
       "beam_size": 1,
       "predictions": "data/ChEMBL/tgt-train",
       "invalid_smiles": True}

with open(opt["targets"], 'r') as f:
    targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

targets = targets[:]
predictions = [[] for i in range(opt["beam_size"])]

test_df = pd.DataFrame(targets)
test_df.columns = ['target']
total = len(test_df)

with open(opt["predictions"], 'r') as f:
    # lines = f.readlines()
    # lines = [''.join(x.strip().split()[1:]) for x in lines]
    # print(lines[1])
    for i, line in enumerate(f.readlines()):
        # if i ==800*10:
        #     break
        predictions[i % opt["beam_size"]].append(''.join(line.strip().split(' ')))

for i, preds in enumerate(predictions):
    test_df['prediction_{}'.format(i + 1)] = preds
    test_df['canonical_prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(
        lambda x: canonicalize_smiles(x))

    properties = {"logP":[], "SA":[], "QED":[], "MW":[]}
    for mol in test_df['canonical_prediction_{}'.format(i + 1)]:
        if mol == "":
            for key in properties:
                properties[key].append(None)
        else:
            m = Chem.MolFromSmiles(mol)
            properties["logP"].append(Descriptors.MolLogP(m))
            properties["SA"].append(sascorer.calculateScore(m))
            properties["QED"].append(Descriptors.qed(m))
            properties["MW"].append(Descriptors.MolWt(m))

    for key in properties:
        test_df['{}_{}'.format(key, i + 1)] = properties[key]

    pickle.dump(properties, open("target_property_{}.pkl".format(i + 1), "wb"))


test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt["beam_size"]), axis=1)



correct = 0
invalid_smiles = 0
for i in range(1, opt["beam_size"] + 1):
    correct += (test_df['rank'] == i).sum()
    invalid_smiles += (test_df['canonical_prediction_{}'.format(i)] == '').sum()
    if opt["invalid_smiles"]:
        print('Top-{}: {:.1f}% || Invalid SMILES {:.2f}%'.format(i, correct / total * 100,
                                                                 invalid_smiles / (total * i) * 100))
    else:
        print('Top-{}: {:.1f}%'.format(i, correct / total * 100))
