import matplotlib.pyplot as plt
import seaborn as sns

n_size = 6

colors = list(sns.color_palette(n_colors=n_size))

custom_legend = []
for i, c in zip(range(len(colors)), colors):
    custom_legend.append(plt.Line2D([], [], color=c, label=f"cls={i}"))
plt.legend(handles=custom_legend)
