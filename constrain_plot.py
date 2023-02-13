import matplotlib.pyplot as plt
import numpy as np


def mark_num_top(x, y, bias=2):
    for i, value in enumerate(y):
        plt.text(x[i], value + bias, value, ha='center', fontsize=13)


labels = ["Equal linker length", "Linker length difference ≤ 1"]
x = np.arange(len(labels))
syntalinker = [67.4, 82.4]
syntalinker_n = [38.3, 62.4]

bar_width = 0.3
n = 2

plt.ylim(0, 120)
plt.xticks(fontsize=13)
plt.ylabel("Percentage (%)", fontsize=13)
plt.bar(x, syntalinker, width=bar_width, label="SyntaLinker", color="#2b3d6b")
plt.bar(x + bar_width, syntalinker_n, width=bar_width, label="SyntaLinker_n", color="#9b9ea3")
plt.xticks(x + bar_width * (n - 1) / 2, labels)

mark_num_top(x, syntalinker)
mark_num_top(x + bar_width, syntalinker_n)

plt.legend(loc='upper right', fontsize=13, frameon=False)
plt.show()
# plt.savefig("constrain.svg", bbox_inches='tight')
