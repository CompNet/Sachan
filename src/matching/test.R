# TODO: Add comment
# 
# Author: Vincent Labatut
# 04/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
###############################################################################


library("igraph")
library("iGraphMatch")




# read the episode-based tvshow static graph
g.tv <- read.graph("in/tvshow/cumul/episode/72.graphml", format="graphml")
# read the scene-based comics static graph
g.cx <- read.graph("in/comics/cumul/scene/cum_1437.graphml", format="graphml")


res <- match_FW <- gm(
	A=g.tv, B=g.cx,			# graphs to compare 
	#seeds,					# known vertex matches
	#similarity,			# vertex-vertex similarity matrix (for method "IsoRank")
	
	method="indefinite",	# default matching method
	start="bari", 			# initialization method for the matrix
	max_iter=200			# maximum number of replacing matches
)

matches <- cbind(res$corr_A, res$corr_B)
print(cbind(V(g.tv)$id[res$corr_A], V(g.cx)$name[res$corr_B]))
