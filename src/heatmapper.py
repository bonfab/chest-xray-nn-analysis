#!/usr/bin/env python

import seaborn as sns; sns.set()
from matplotlib import cm
import matplotlib.pyplot as plt

def visualize(img, heatmap, explanation=''):

  f, axes = plt.subplots(1, 3,  figsize=[12, 3])

  axes[0].title.set_text('Image')
  axes[0].imshow(img)
  axes[0].grid(None)

  axes[1].title.set_text('Heatmap')
  axes[1].imshow(heatmap)
  axes[1].grid(None)

  illustration = sns.heatmap(heatmap,
                  xticklabels=False,
                  yticklabels=False,
                  alpha=0.5,
                  zorder = 2,
                  cmap = cm.PuBuGn,
                  cbar_kws={'label': 'Intensity'},
                  ax=axes[2])
  illustration.imshow(img,
           aspect = illustration.get_aspect(),
           extent = illustration.get_xlim() + illustration.get_ylim(),
           zorder = 1)

  axes[2].title.set_text(f"Heatmap + Image {explanation}")
  plt.show()