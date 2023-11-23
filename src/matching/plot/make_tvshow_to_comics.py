# -*- eval: (code-cells-mode); -*-

# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from alignment_commons import TVSHOW_SEASON_LIMITS, NOVEL_LIMITS

with open("./tvshow_novels_gold_alignment.pickle", "rb") as f:
    M_tn = pickle.load(f)
M_tn = M_tn[: TVSHOW_SEASON_LIMITS[1], : NOVEL_LIMITS[1]]

with open("./novels_comics_gold_alignment.pickle", "rb") as f:
    M_nc = pickle.load(f)


# M_tc = np.zeros((M_tn.shape[0], M_nc.shape[1]))
M_tc = M_tn @ M_nc
fig, axs = plt.subplots(3, 1)
axs[0].imshow(M_tn)
axs[1].imshow(M_nc)
axs[2].imshow(M_tc)
plt.imshow(M_tc)
plt.show()


# %%
with open("./tvshow_comics_gold_alignment.pickle", "wb") as f:
    pickle.dump(M_tc, f)
