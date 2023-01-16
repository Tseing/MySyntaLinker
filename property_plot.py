import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

output = pickle.load(open("out_property_1.pkl", 'rb'))
target = pickle.load(open("target_property_1.pkl", 'rb'))
for key in output:
    output[key] = list(filter(None, output[key]))

output_f = {key: None for key in output}
for key in output_f:
    output_f[key] = gaussian_kde(output[key])

properties = [key for key in output]
index = 0
row, col = 2, 2
fig, subs=plt.subplots(row, col)
plt.subplots_adjust(wspace=0.55, hspace=0.3)

for i in range(row):
    for j in range(col):
        property = properties[index]
        _, out_bins, _ = subs[i][j].hist(output[property],
                        bins=200, density=True, alpha=0.3, color="C2")

        _, tgt_bins, _ = subs[i][j].hist(target[property],
                        bins=200, density=True, alpha=0.3, color="C0")

        subs[i][j].plot(out_bins[:-1], output_f[property](out_bins[:-1]),
                        color="C2", label="ChEMBL Set")
        subs[i][j].plot(tgt_bins[:-1], output_f[property](tgt_bins[:-1]),
                        color="C0", label="SyntaLinker")
        subs[i][j].set_xlabel(property, fontsize=11)
        subs[i][j].legend(fontsize=6, frameon=False)

        index += 1

plt.show()
# plt.savefig("distribution.svg")
