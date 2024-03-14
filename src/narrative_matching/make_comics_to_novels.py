# -*- eval: (code-cells-mode); -*-

# %%
# Produce comics-novels gold alignment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alignment_commons import NOVEL_LIMITS

ISSUES_NB = 56
M_cn = np.zeros((ISSUES_NB, NOVEL_LIMITS[1]))

# CSV mapping exported from asoiaf_comics
#
# format:
# Volume,StartPage,EndPage,Chapter,Rank,PoV,Number
#
# example line:
# AGOT01,3,9,0,1,Prologue,I
df = pd.read_csv("./asoiaf_comics_mapping.csv")

arc_offsets = {"AGOT": 0, "ACOK": 24}

for _, row in df.iterrows():
    arc = row["Volume"][:4]
    issue_i = arc_offsets[arc] + int(row["Volume"][4:]) - 1
    chapter_i = row["Rank"] - 1
    M_cn[issue_i][chapter_i] = 1


plt.imshow(M_cn)
plt.show()


# %%
# Export alignment
import os, sys, pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f"{script_dir}/../../.."
sys.path.append(f"{root_dir}/src")

with open(
    f"{root_dir}/in/plot_alignment/comics_novels_i2c_gold_alignment.pickle", "wb"
) as f:
    pickle.dump(M_cn, f)
