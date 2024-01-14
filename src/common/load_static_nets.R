# Loads static networks and orders characters by importance, using their degree 
# in all three static networks. The edge weights are max-normalized for each
# network, in order to get comparable values.
# 
# Author: Vincent Labatut
# 08/2023
#
# setwd("C:/Users/Vincent/eclipse/workspaces/Networks/Sachan")
# source("src/common/load_static_nets.R")
###############################################################################




###############################################################################
# parameters
{	if(NARRATIVE_PART==0)
	{	file.nv <- "in/novels/cumul/5.ADwD_72_cumul.graphml"
		file.cx <- "in/comics/cumul/scene/cum_1437.graphml"
		file.tv <- "in/tvshow/cumul/scene/cumulative_4164.graphml"
	}
	else if(NARRATIVE_PART==2)
	{	file.nv <- "in/novels/cumul/2.ACoK_69_cumul.graphml"
		file.cx <- "in/comics/cumul/scene/cum_1437.graphml"
		file.tv <- "in/tvshow/cumul/scene/cumulative_0753.graphml"
	}
	else if(NARRATIVE_PART==5)
	{	file.nv <- "in/novels/cumul/5.ADwD_72_cumul.graphml"
		file.cx <- NA
		file.tv <- "in/tvshow/cumul/scene/cumulative_2248.graphml"
	}
}




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
g.nv <- read.graph(file.nv, format="graphml")
g.nv <- delete_vertices(graph=g.nv, v=!V(g.nv)$named)				# keep only named characters
E(g.nv)$weight <- E(g.nv)$weight/max(E(g.nv)$weight)				# normalize weights

# read the scene-based comics static graph
if(!is.na(file.cx))
{	g.cx <- read.graph(file.cx, format="graphml")
	g.cx <- delete_vertices(graph=g.cx, v=!V(g.cx)$named)			# keep only named characters
	E(g.cx)$weight <- E(g.cx)$Occurrences/max(E(g.cx)$Occurrences)	# normalize weights
}

# read the episode-based tvshow static graph
g.tv <- read.graph(file.tv, format="graphml")
g.tv <- delete_vertices(graph=g.tv, v=!V(g.tv)$named)				# keep only named characters
E(g.tv)$weight <- E(g.tv)$weight/max(E(g.tv)$weight)				# normalize weights




###############################################################################
# retrieve the characters' affiliations
char.file <- "in/characters.csv"
char.tab <- read.csv2(char.file, header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
# clean up a bit
aff.map <- char.tab[,"AllegianceBoth"]
names(aff.map) <- char.tab[,"Name"]
aff.map[aff.map==""] <- "Unknown"
aff.map <- sapply(strsplit(x=aff.map, split=",", fixed=TRUE), function(v) v[1])

# add to novel network
aff <- aff.map[V(g.nv)$name]
aff[is.na(aff)] <- "Unknown"
V(g.nv)$affiliation <- aff
# add to comics network
if(!is.na(file.cx))
{	aff <- aff.map[V(g.cx)$name]
	aff[is.na(aff)] <- "Unknown"
	V(g.cx)$affiliation <- aff
}
# add to TV show network
aff <- aff.map[V(g.tv)$name]
aff[is.na(aff)] <- "Unknown"
V(g.tv)$affiliation <- aff




###############################################################################
# list of graphs
{	if(!is.na(file.cx))
		gs <- list("novels"=g.nv, "comics"=g.cx, "tvshow"=g.tv)
	else
		gs <- list("novels"=g.nv, "tvshow"=g.tv)
	g.names <- names(gs)
}




###############################################################################
## compute a list of characters ranked by importance
#all.char.names <- sort(unique(unlist(sapply(gs, function(g) V(g)$name))))
#imp.mat <- matrix(NA, nrow=length(all.char.names), ncol=length(gs))
#rownames(imp.mat) <- all.char.names
#colnames(imp.mat) <- g.names
#for(i in 1:length(gs))
#	# we use the degree as a proxy for importance
#	imp.mat[match(V(gs[[i]])$name, all.char.names),g.names[i]] <- degree(gs[[i]])/gorder(gs[[i]])
#imp.moy <- apply(imp.mat,1,function(v) mean(v,na.rm=TRUE))
#ranked.chars <- all.char.names[order(imp.moy,decreasing=TRUE)]
#
## export for later use
#char.importance <- data.frame(all.char.names,imp.mat,imp.moy)
#char.importance <- char.importance[order(imp.moy,decreasing=TRUE),]
#rownames(char.importance) <- NULL
#colnames(char.importance) <- if(NARRATIVE_PART<5) c("Name","Novels","Comics","TVshow","Mean") else c("Name","Novels","TVshow","Mean")
#write.csv(x=char.importance, file=file.path("in",paste0("ranked_importance_S",NARRATIVE_PART,".csv")), row.names=FALSE, fileEncoding="UTF-8")

# 0: "Tyrion Lannister" "Jon Snow"	    "Theon Greyjoy" "Arya Stark"      "Sansa Stark"   "Catelyn Stark"
# 2: "Tyrion Lannister" "Catelyn Stark" "Theon Greyjoy" "Eddard Stark"    "Arya Stark"    "Joffrey Baratheon"
# 5: "Tyrion Lannister" "Jon Snow"      "Arya Stark"    "Jaime Lannister" "Catelyn Stark" "Sansa Stark"

# note: above code obsolete, process now performed in the below script 
source("src/common/char_importance.R")
