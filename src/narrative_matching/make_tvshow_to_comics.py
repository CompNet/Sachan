# -*- eval: (code-cells-mode); -*-

# %%
import numpy as np
import matplotlib.pyplot as plt
from alignment_commons import (
    TVSHOW_SEASON_LIMITS,
    NOVEL_LIMITS,
    load_medias_gold_alignment,
)


M_tn = load_medias_gold_alignment("tvshow-novels")
M_tn = M_tn[: TVSHOW_SEASON_LIMITS[1], : NOVEL_LIMITS[1]]
M_cn = load_medias_gold_alignment("comics-novels")


M_tc = M_tn @ M_cn.T
fig, axs = plt.subplots(3, 1)
axs[0].set_title("M_tn")
axs[0].imshow(M_tn)
axs[1].set_title("M_cn")
axs[1].imshow(M_cn)
axs[2].set_title("M_tc")
axs[2].imshow(M_tc)
plt.imshow(M_tc)
plt.show()


# %%
import os, sys, pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../.."
sys.path.append(f"{root_dir}/src")

M_tc[M_tc > 1] = 1
with open(
    f"{root_dir}/in/narrative_matching/tvshow_comics_i2e_gold_alignment.pickle", "wb"
) as f:
    pickle.dump(M_tc, f)
