import matplotlib.pyplot as plt


x = ["Equal linker length", "Linker length difference ≤ 1"]
y = [93.7, 97.0]

plt.ylim(0, 120)
plt.xticks(fontsize=13)
plt.ylabel("Percentage (%)", fontsize=13)
plt.bar(x, y, width=0.3, label="SyntaLinker", color="#2b3d6b")

for x, value in enumerate(y):
    plt.text(x, value+2, value, ha='center', fontsize=13)
plt.legend(loc='upper right', fontsize=13, frameon=False)
plt.show()
# plt.savefig("constrain.svg")
