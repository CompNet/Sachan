# Lists the names used in the different media, in order to normalize them.
# Used only once, to prepare the name conversion maps.
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/name_normalization.R")
###############################################################################
library("igraph")




###############################################################################
# read the static tvshow graph
g.tv <- read.graph("in/tvshow/cumul/episode/71.graphml", format="graphml")
names <- sort(V(g.tv)$name)
tab.file <- "in/tvshow/charlist.csv"
write.csv(x=names, file=tab.file, row.names=FALSE, fileEncoding="UTF-8")




###############################################################################
# read the static novel graph
g.nv <- read.graph("in/novels/cumul/5.ADwD_72_cumul.graphml", format="graphml")
names <- sort(V(g.nv)$id)
tab.file <- "in/novels/charlist.csv"
write.csv(x=names, file=tab.file, row.names=FALSE, fileEncoding="UTF-8")




###############################################################################
# comics already have been normalized


