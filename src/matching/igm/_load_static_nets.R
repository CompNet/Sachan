# Loads static networks and orders characters by importance, using their degree 
# in all three static networks.
# 
# Author: Vincent Labatut
# 08/2023
###############################################################################




###############################################################################
# load the static graphs

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
g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)			# keep only named characters
E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)			# normalize weights

# read the scene-based comics static graph
g.cx <- read.graph("in/comics/cumul/scene/cum_1437.graphml", format="graphml")
g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)			# keep only named characters
E(g.cx)$weight <- E(g.cx)$Occurrences/max(E(g.cx)$Occurrences)	# normalize weights

# read the episode-based tvshow static graph
g.tv <- read.graph("in/tvshow/cumul/scene/cumulative_0753.graphml", format="graphml")
g.tv <- delete_vertices(graph=g.tv, v=!V(g.tv)$named)			# keep only named characters
E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)			# normalize weights




###############################################################################
# rank characters using degree in each network
names <- sort(union(V(g.nv)$name,union(V(g.cx)$name,V(g.tv)$name)))
imp.mat <- matrix(NA, nrow=length(names), ncol=3)
rownames(imp.mat) <- names
colnames(imp.mat) <- c("novels","comics","tvshow")
imp.mat[match(V(g.nv)$name, names),"novels"] <- degree(g.nv)/gorder(g.nv)
imp.mat[match(V(g.cx)$name, names),"comics"] <- degree(g.cx)/gorder(g.cx)
imp.mat[match(V(g.tv)$name, names),"tvshow"] <- degree(g.tv)/gorder(g.tv)
imp.moy <- apply(imp.mat,1,function(v) mean(v,na.rm=TRUE))
ranked.chars <- names[order(imp.moy,decreasing=TRUE)]
