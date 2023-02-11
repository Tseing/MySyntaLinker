import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

output = pickle.load(open("property_SyntaLinker_top10.pkl", 'rb'))
output_n = pickle.load(open("property_SyntaLinker_n_top10.pkl", 'rb'))
target = pickle.load(open("property_ChEMBL.pkl", 'rb'))


def is_value(item):
    if item == "" or item is None or np.isnan(item):
        return False
    else:
        return True


def get_gaussian_ked(data_dict: dict):
    f = {key: None for key in data_dict}
    for key in properties:
        data_dict[key] = np.array(list(filter(is_value, data_dict[key])))
        f[key] = gaussian_kde(data_dict[key])

    return f


properties = [key for key in target]
index = 0
row, col = 2, 2
fig, subs = plt.subplots(row, col)
plt.subplots_adjust(wspace=0.55, hspace=0.3)

for i in range(row):
    for j in range(col):
        property = properties[index]
        _, out_bins, _ = subs[i][j].hist(output[property], bins=200,
                                         density=True, alpha=0.3, color="C2")
        _, out_n_bins, _ = subs[i][j].hist(output_n[property], bins=200,
                                           density=True, alpha=0.3, color="C1")
        _, tgt_bins, _ = subs[i][j].hist(target[property], bins=200,
                                         density=True, alpha=0.3, color="C0")

        subs[i][j].plot(out_bins[:-1], get_gaussian_ked(target)[property](out_bins[:-1]),
                        color="C2", label="ChEMBL Set")
        subs[i][j].plot(out_n_bins[:-1], get_gaussian_ked(output_n)[property](out_n_bins[:-1]),
                        color="C1", label="SyntaLinker_n")
        subs[i][j].plot(tgt_bins[:-1], get_gaussian_ked(output)[property](tgt_bins[:-1]),
                        color="C0", label="SyntaLinker")
        subs[i][j].set_xlabel(property, fontsize=11)
        subs[i][j].legend(fontsize=6, frameon=False)

        index += 1

plt.show()
# plt.savefig("distribution.svg")
