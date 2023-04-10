# TODO: Add comment
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/matching/test.R")
###############################################################################
library("igraph")
library("iGraphMatch")




###############################################################################
# notes:
# - comics:
#   - last chapter: #143 (file cum/inst_143.graphml)
#   - last scene: #1438 (file cum/inst_1437.graphml)
# - tv show:
#   - last S02 episode: #20 (file cumulative/instant_019.graphml)
#   - last S02 scene: #754 (file cumulative/instant_0753.graphml)
#   - very last episode: #73 (file cumulative/instant_072.graphml)
#   - very last scene: #4165 (file cumulative/instant_4164.graphml)
# - novels:
#   - last book 2 chapter: 2.ACoK_69_cumul.graphml / 2.ACoK_69_instant.graphml
#   - last chapter: 5.ADwD_72_cumul.graphml / 5.ADwD_72_instant.graphml

# read the chapter-based novel static graph
g.nv <- read.graph("in/novels/cumul/2.ACoK_69_cumul.graphml", format="graphml")
E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)
# read the episode-based tvshow static graph
g.tv <- read.graph("in/tvshow/cumul/scene/cumulative_0753.graphml", format="graphml")
E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)
# read the scene-based comics static graph
g.cx <- read.graph("in/comics/cumul/scene/cum_1437.graphml", format="graphml")
E(g.cx)$weight <- E(g.cx)$Occurrences/max(E(g.cx)$Occurrences)




###############################################################################
res <- match_FW <- gm(
	A=g.nv, B=g.cx,			# graphs to compare 
	#seeds,					# known vertex matches
	#similarity,			# vertex-vertex similarity matrix (for method "IsoRank")
	
	method="indefinite",	# default matching method
	start="bari", 			# initialization method for the matrix
	max_iter=200			# maximum number of replacing matches
)

matches <- cbind(res$corr_A, res$corr_B)
print(cbind(V(g.tv)$name[res$corr_A], V(g.cx)$name[res$corr_B]))
