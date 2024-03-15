# Main script for the processing implemented in R.
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/main.R")
###############################################################################
library("igraph")
library("iGraphMatch")
library("viridis")
library("XML")
library("plot.matrix")
library("scales")
library("fmsb")
library("cluster")
library("latex2exp")
library("SDMTools")




###############################################################################
source("src/visualization/narr_overlap.R")

# plot static graphs
source("src/visualization/static_plots.R")

# plot top-20 graphs
source("src/visualization/top20_plots.R")




###############################################################################
# topological properties of static graphs
source("src/descript/static_props.R")

# evolution of topological measures
source("src/descript/evol_props.R")




###############################################################################
# graph matching
source("src/vertex_matching/igm/exp_simple.R")
source("src/vertex_matching/igm/exp_adaptive_hard_seeds.R")
source("src/vertex_matching/igm/exp_adaptive_hard_seeds_temporal.R")
source("src/vertex_matching/igm/exp_adaptive_soft_seeds.R")

# jaccard sim matching
source("src/vertex_matching/jaccard/descriptive_static.R")
source("src/vertex_matching/jaccard/descriptive_dynamic.R")
source("src/vertex_matching/jaccard/specific_chars.R")

# centrality study
source("src/descript/char_centrality.R")
