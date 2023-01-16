import os
import re

data_path = "data/ChEMBL"
data_names = ["src-test.txt", "src-train", "src-val"]
save_path = "data/SyntaLinker_n"


def del_SLBD(line: str):
    replaced_line = re.sub(r"L_[0-9]+\s", "", line)
    return replaced_line


if not os.path.exists(save_path):
    os.mkdir(save_path)

for name in data_names:
    with open(os.path.join(data_path, name), "r") as f:
        with open(os.path.join(save_path, name), "w") as s:
            for line in f:
                s.write(del_SLBD(line))
