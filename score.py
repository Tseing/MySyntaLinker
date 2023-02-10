import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import moses.metrics.metrics as metrics
from prettytable import PrettyTable


MODEL = "SyntaLinker_top10"

opt = {"targets": f"data/{MODEL}/tgt-test.txt",
       "beam_size": 10,
       "predictions": f"data/{MODEL}/output.txt",
       "score_top_n": True,
       "score_linker": True
       }

# constrained and unconstrained
# 有 linker 长度约束信息的文件与无约束信息有 linker 的文件
test_file = {"cons": f"data/{MODEL}/src-test.txt",
             "uncons": opt["predictions"], }

train_file = {"cons": f"data/shared/src-train",
              "uncons": f"data/shared/tgt-train"}


def get_rank(row, base: str, max_rank: int) -> int:
    # 在 rank 一列存储 top-n 个结果中正确结果的索引 n (求秩)
    for i in range(1, max_rank + 1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def is_legal_mol(smiles: str) -> bool:
    # RDkit 导出的 canonical smiles 也可能有错，使用此函数确保安全
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return True

    return False


def get_properties_from(data: DataFrame, column_name: str, column_num: int):
    properties = {"logP": None, "SA": None, "QED": None, "MW": None}
    for k in properties.keys():
        properties[k] = [[] for i in range(column_num)]

    for i in range(column_num):
        for mol in data[f'{column_name}_{i + 1}']:
            if not is_legal_mol(mol):
                for k in properties.keys():
                    properties[k][i].append(float('nan'))
            else:
                m = Chem.MolFromSmiles(mol)
                properties["logP"][i].append(Descriptors.MolLogP(m))
                properties["SA"][i].append(sascorer.calculateScore(m))
                properties["QED"][i].append(Descriptors.qed(m))
                properties["MW"][i].append(Descriptors.MolWt(m))

    for k in properties.keys():
        properties[k] = np.nanmean(properties[k], axis=0)

    return properties


def get_linker(smi: str, patt1: str, patt2: str):
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


def remove_debris_atom(smi: str) -> str:
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


def is_dual_site(linker: str) -> bool:
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


def read_src_tgt_data(dataset: str) -> DataFrame:
    # 生成类似 src-tgt (cons-uncons) 的对照数据表
    data = pd.DataFrame()
    if dataset == "test":
        file = test_file
        fill_data_by_column(file['uncons'], data, "prediction")
    elif dataset == "train":
        file = train_file
        fill_data_by_column(file['uncons'], data, "mol", column_num=1)
    else:
        raise ValueError

    return data


def generate_constraint(data: DataFrame, dataset: str) -> None:
    if dataset == "test":
        file = test_file
    elif dataset == "train":
        file = train_file
    else:
        raise ValueError

    pairs, constraint = [], []
    with open(file["cons"], 'r') as f:
        for line in f:
            pairs.append(re.sub(r"L_\d+", "", ''.join(line.strip().split(' '))))
            constraint.append(int(re.search(r"\d+", re.search(r"L_\d+", line).group()).group()))

    data['pairs'] = pairs
    data['constraint'] = constraint


def generate_linker_from(data: DataFrame, column_name: str, column_num=opt["beam_size"]) -> None:
    def append_linker(mol, pairs_idx):
        if mol:
            patt = tuple(data['pairs'][pairs_idx].split("."))
            linker = get_linker(mol, *patt)
            if is_dual_site(linker):
                linkers.append(linker)
            else:
                linkers.append("")
        else:
            linkers.append("")

    linkers_data = [[] for i in range(column_num)]

    for i, linkers in enumerate(linkers_data):
        idx = 0
        for mol in data[f'{column_name}_{i + 1}']:
            append_linker(mol, idx)
            idx += 1
        data[f'linker_{i + 1}'] = linkers


def calculate_SLBD(data: DataFrame, column_num=opt["beam_size"]):
    # 计算 linker 是否满足训练集中的 SLBD 约束
    def calculate_linker_length():
        linker_lens = [[] for i in range(column_num)]
        for i, linker_len in enumerate(linker_lens):
            for linker in data[f'linker_{i + 1}']:
                if linker:
                    linker_mol = Chem.MolFromSmiles(linker)
                    linker_site_idxs = [atom.GetIdx() for atom in linker_mol.GetAtoms() if atom.GetAtomicNum() == 0]
                    linker_len.append(int(len(Chem.rdmolops.GetShortestPath(linker_mol, linker_site_idxs[0], linker_site_idxs[1])) - 2))
                else:
                    # 正常 linker 长度不可能为 0，代表错误 linker，0 比 None 更好用于计算
                    linker_len.append(0)

        return np.array(linker_lens)

    linker_lens = calculate_linker_length()
    SLBD_bools = [linker_len == np.array(data['constraint']) for linker_len in linker_lens]
    SLBD = np.sum(SLBD_bools, axis=1)
    SLBD_leq_bools = [(np.abs(linker_len - np.array(data['constraint'])) <= 1)
                      for linker_len in linker_lens]
    SLBD_leq = np.sum(SLBD_leq_bools, axis=1)

    return SLBD, SLBD_leq


def fill_data_by_column(file: str, table, column_name: str, column_num=opt["beam_size"]) -> None:
    # 根据 beam_size 将分子数分为 n 组并正则化
    columns = [[] for i in range(column_num)]
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            columns[i % column_num].append(''.join(line.strip().split(' ')))
    if column_num < 1:
        raise ValueError
    else:
        for i, column in enumerate(columns):
            table[f'{column_name}_{i + 1}'] = column
            table[f'canonical_{column_name}_{i + 1}'] = table[f'{column_name}_{i + 1}'].apply(
                lambda x: canonicalize_smiles(x))


def score(test_df: DataFrame, train_df: DataFrame, linker_df: DataFrame, dataset: str):
    scores = {}
    valids = []
    uniques = []
    novelties = []
    mols_top_n = []

    if dataset == "linker":
        train_data = train_df['linker_1']
        process_df = linker_df
        process_column = "linker"

    elif dataset == "molecule":
        train_data = train_df['mol_1']
        process_df = test_df
        process_column = "canonical_prediction"

        corrects = [(test_df['rank'] == i).sum() for i in range(top_n)]
        scores["recovery"] = [sum(corrects[:i + 1]) / (total * (i + 1)) * 100 for i in range(top_n)]
        scores["SLBD"] = [sum(SLBD[:i + 1] / (total * (i + 1)) * 100) for i in range(top_n)]
        scores["SLBD_leq"] = [sum(SLBD_leq[:i + 1] / (total * (i + 1)) * 100) for i in range(top_n)]

    else:
        raise ValueError

    for i in range(top_n):
        # 第 i 列分子
        mols = list(filter(is_legal_mol, process_df[f'{process_column}_{i + 1}'].tolist()))
        # 前 i 列分子
        mols_top_n += mols
        valids.append(metrics.fraction_valid(mols, n_jobs=2))
        uniques.append(metrics.fraction_unique(mols_top_n, 1000,  n_jobs=2))
        novelties.append(metrics.novelty(mols, train_data,  n_jobs=2))

    scores['validity'] = [sum(valids[:i + 1]) / (i + 1) * 100 for i in range(top_n)]
    scores['uniqueness'] = [num * 100 for num in uniques]
    scores['novelty'] = [sum(novelties[:i + 1]) / (i + 1) * 100 for i in range(top_n)]

    return scores


if __name__ == "__main__":
    with open(opt["targets"], 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    targets = targets[:]
    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    total = len(test_df)

    fill_data_by_column(opt["predictions"], test_df, "prediction")
    test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt["beam_size"]), axis=1)

    linker_df = read_src_tgt_data("test")
    generate_constraint(linker_df, dataset="test")
    generate_linker_from(linker_df, "canonical_prediction")
    SLBD, SLBD_leq = calculate_SLBD(linker_df)

    if not os.path.exists("./train_df.pkl"):
        train_df = read_src_tgt_data("train")
        generate_constraint(train_df, dataset="train")
        generate_linker_from(train_df, "mol", column_num=1)
        pickle.dump(train_df, open(f"train_df.pkl", "wb"))
    else:
        train_df = pickle.load(open(f"train_df.pkl", "rb"))

    if not os.path.exists(f"./property_{MODEL}.pkl"):
        properties = get_properties_from(test_df, 'canonical_prediction', opt["beam_size"])
        pickle.dump(properties, open(f"property_{MODEL}.pkl", "wb"))

    if not os.path.exists("./property_ChEMBL.pkl"):
        ChEMBL_properties = get_properties_from(train_df, 'mol', column_num=1)
        pickle.dump(ChEMBL_properties, open("property_ChEMBL.pkl", "wb"))

    # top_n 为指定输出前 n 列的评估数据，否则输出全部（所有列整合在一起评估）
    if opt["score_top_n"]:
        top_n = opt["beam_size"]
    else:
        top_n = 1

    scores = score(test_df, train_df, linker_df, "molecule")

    title = [f"Top_{i + 1}" for i in range(top_n)]
    table = PrettyTable(["Molecule Metrics"] + title, float_format=".2")
    table.add_row(["Recovery"] + scores['recovery'])
    table.add_row(["Validity"] + scores['validity'])
    table.add_row(["Uniqueness"] + scores['uniqueness'])
    table.add_row(["Novelty"] + scores['novelty'])
    table.add_row(["SLBD"] + scores['SLBD'])
    table.add_row(["SLBD (leq 1)"] + scores['SLBD_leq'])

    print(table)

    if opt["score_linker"]:
        scores = score(linker_df, train_df, linker_df, "linker")

        linker_table = PrettyTable(["Linker Metrics"] + title, float_format=".2")
        linker_table.add_row(["Validity"] + scores['validity'])
        linker_table.add_row(["Uniqueness"] + scores['uniqueness'])
        linker_table.add_row(["Novelty"] + scores['novelty'])

        print(linker_table)

    print("")
