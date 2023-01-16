import re
import os
import pickle
import moses
import pandas as pd
from rdkit import Chem

test_file = {"src": "data/ChEMBL/src-test.txt",
             "tgt": "output.txt",}

train_file = {"src": "data/ChEMBL/src-train",
              "tgt": "data/ChEMBL/tgt-train"}


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def get_linker(smi, patt1, patt2):
    mol = Chem.MolFromSmiles(smi)
    # 使用 RWMol 编辑后索引发生变化，影响分割，必须再转换一次
    patt1_smi = remove_extra_bond(Chem.MolFromSmarts(patt1))
    patt2_smi = remove_extra_bond(Chem.MolFromSmarts(patt2))
    patt1 = Chem.MolFromSmarts(patt1_smi)
    patt2 = Chem.MolFromSmarts(patt2_smi)
    try:
        # 无法分割时先尝试调换片段分割次序
        linker = Chem.ReplaceCore(Chem.ReplaceCore(mol, patt1), patt2)
        if linker is None:
            linker = Chem.ReplaceCore(Chem.ReplaceCore(mol, patt2), patt1)
    except:
        return False

    if linker is not None:
        linker_smi = Chem.MolToSmiles(linker)
        return remove_debris_atom(linker_smi)
    else:
        return False


def remove_debris_atom(smi):
    # 清除分割后可能生成的零星片段
    smi = re.sub(r"\[\d+\*\]", "[*]", smi)
    fragments = smi.split(".")
    if len(fragments) != 1:
        lens = [len(frag) for frag in fragments]
        frags = dict(zip(lens, fragments))
        return frags[max(lens)]
    else:
        return smi


def remove_extra_bond(mol):
    # 删除母核中与 [*] 相连的化学键，否则会额外减少 linker 的长度
    site = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    site_idx = site[0]
    edit_m = Chem.RWMol(mol)
    neighbor_atom = mol.GetAtomWithIdx(site_idx).GetNeighbors()[0]
    edit_m.ReplaceAtom(neighbor_atom.GetIdx(), Chem.Atom(0))
    edit_m.RemoveBond(site_idx, neighbor_atom.GetIdx())
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            edit_m.RemoveAtom(atom.GetIdx())

    edit_m = edit_m.GetMol()
    return Chem.MolToSmiles(edit_m)


def is_dual_site(linker):
    # 若分割得到的 linker 不具有两个 [*] 位点则视作非法
    try:
        linker_mol = Chem.MolFromSmiles(linker)
    except:
        return False
    if linker_mol is not None and \
            len([atom.GetIdx() for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0]) == 2:
        return True
    else:
        return False


def generate_linker_and_constraint(dataset="test"):
    if dataset == "test":
        file = test_file
        data = pd.DataFrame()
    elif dataset == "train":
        file = train_file
        data = pd.DataFrame()
    else:
        raise ValueError

    src, constraint = [], []
    with open(file["src"], 'r') as f:
        for line in f:
            src.append(re.sub(r"L_\d+", "", ''.join(line.strip().split(' '))))
            constraint.append(re.search(r"\d+", re.search(r"L_\d+", line).group()).group())

    with open(file["tgt"], 'r') as f:
        tgt = [''.join(line.strip().split(' ')) for line in f]

    data = pd.DataFrame(src)
    data.columns = ['src']
    data['tgt'] = tgt
    data['canonical'] = data['tgt'].apply(lambda x: canonicalize_smiles(x))

    idx = 0
    error_linker_cnt = 0
    linkers = []
    for mol in data['canonical']:
        if mol:
            patt = tuple(data['src'][idx].split("."))
            linker = get_linker(mol, *patt)
            if is_dual_site(linker):
                linkers.append(linker)
            else:
                linkers.append("")
                error_linker_cnt += 1

        else:
            linkers.append("")

        idx += 1

    data['linkers'] = linkers
    data['constrain'] = constraint

    return data


def calculate_SLBD(table):
    # 计算 linker 是否满足训练集中的 SLBD 约束
    linker_lens = []
    SLBD = []
    SLBD_leq = []
    idx = 0
    for linker in table['linkers']:
        if linker:
            linker_mol = Chem.MolFromSmiles(linker)
            linker_site_idxs = [atom.GetIdx() for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0]
            linker_len = len(Chem.rdmolops.GetShortestPath(linker_mol, linker_site_idxs[0], linker_site_idxs[1])) - 2
            linker_lens.append(linker_len)

            if linker_len == int(table['constrain'][idx]):
                SLBD.append(1)
            else:
                SLBD.append(0)

            if linker_len == int(table['constrain'][idx]) or\
                (linker_len - int(table['constrain'][idx]) == 1) or (int(table['constrain'][idx]) - linker_len == 1):
                SLBD_leq.append(1)
            else:
                SLBD_leq.append(0)
        else:
            linker_lens.append(None)
            SLBD.append(0)
            SLBD_leq.append(0)

        idx += 1

    table['linker_lens'] = linker_lens
    table['SLBD'] = SLBD
    table['SLBD_leq'] = SLBD_leq


if __name__ == "__main__":
    test = generate_linker_and_constraint("test")
    # calculate_SLBD(test)
    if not os.path.exists("train_linker.pkl"):
        train = generate_linker_and_constraint("train")
        pickle.dump(train, open("train_linker.pkl", "wb"))
    else:
        train = pickle.load(open("train_linker.pkl", "rb"))

    valid = len(list(filter(lambda x: x != "", list(test['linkers']))))
    # valid = len(list(filter(lambda x: x != "", list(test['canonical']))))
    total = len(test['canonical'])
    validity = valid / total
    non_duplicate = len(set(test['linkers'])) - 1
    # non_duplicate = len(set(test['canonical'])) - 1
    uniqueness = non_duplicate / valid
    novel_linker = [1 for linker in set(test['linkers']) if linker not in set(train['linkers'])]
    novel = [True for linker in set(test['canonical']) if linker not in set(train['canonical'])]
    novelty = len(novel) / non_duplicate
    fulfill_SLBD = sum(test['SLBD']) / valid
    fulfill_SLBD_leq = sum(test['SLBD_leq']) / valid

    # metrics = moses.get_all_metrics(test['tgt'].tolist(), train=train['tgt'].tolist())

    # print(metrics)
    print("linker validity: {}".format(validity))
    print("uniqueness: {}".format(uniqueness))
    print("novelty: {}".format(novelty))
    print("fulfill: {}".format(fulfill_SLBD))
    print("fulfill_leq_1: {}".format(fulfill_SLBD_leq))
    print("")
